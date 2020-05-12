#![allow(dead_code, unused_imports, unused_variables, unused_parens, unstable_name_collisions)]

use std::mem;
use std::ops::{Add, Mul, Sub};
use std::ptr;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::SystemTime;

use imgui::*;
use minifb::{Key, Window, WindowOptions};
use ultraviolet::vec::{Vec2, Vec3, Vec4};
use winapi::um::xinput::XINPUT_GAMEPAD_A;
use winapi;

const WIDTH: u32 = 1920;
const WIDTH_F: f32 = WIDTH as f32;
const WIDTH_USIZE: usize = WIDTH as usize;
const HEIGHT: u32 = 1080;
const HEIGHT_F: f32 = HEIGHT as f32;
const HEIGHT_USIZE: usize = HEIGHT as usize;

const CLEAR_COLOR: u32 = 0x0000_0000;

fn y_axis_rotate(p: Vec4, theta: f32) -> Vec4 {
    Vec4::new(
        p.x*theta.cos() - p.z*theta.sin(),
        p.y,
        p.x*theta.sin() + p.z*theta.cos(),
        1.0f32,
    )
}

struct LineSegment {
    a: Vec2,
    b: Vec2,
}

impl LineSegment {
    fn new(a: Vec2, b: Vec2) -> Self {
        Self {a, b}
    }

    /// For a given y, how far are are we along the line segment (in terms of
    /// x position and ratio along the line)
    fn x_for_y(&self, y: f32) -> Option<f32> {
        let dx = self.b.y - self.a.y;
        if dx == 0f32 {
            // There's no t for this
            None
        } else {
            let t = ((y as f32) - self.a.y) / dx;
            if 0.0f32 <= t && t <= 1.0f32 {
                Some(((self.b - self.a) * t + self.a).x)
            } else {
                None
            }
        }
    }
}

#[derive(Copy, Clone)]
struct Vert {
    pos: Vec4,
    col: Vec3,
    uv: Vec2,
}

impl Vert {
    fn new_from_imgui(pos: Vec4, col: [u8; 4], uv: [f32; 2]) -> Self {
        let r = col[0] as f32 / 255f32;
        let g = col[1] as f32 / 255f32;
        let b = col[2] as f32 / 255f32;

        let col = Vec3::new(r,g,b);
        let uv = Vec2::new(uv[0], uv[1]);
        Self {pos, col, uv}
    }

    fn newf32(pos: Vec4, col: Vec3, uv: Vec2) -> Self {
        Self {pos, col, uv}
    }
}

#[derive(Copy, Clone, Debug)]
struct Rect {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
}

impl Rect {
    fn new(x0: f32, x1: f32, y0: f32, y1: f32) -> Self {
        Self {x0, x1, y0, y1}
    }

    fn intersect(&self, other: &Self) -> Self {
        let x0 = f32::max(self.x0, other.x0);
        let x1 = f32::min(self.x1, other.x1);
        let y0 = f32::max(self.y0, other.y0);
        let y1 = f32::min(self.y1, other.y1);

        Self {x0, x1, y0, y1}
    }
}

fn run_render_thread(
    input_channel: mpsc::Receiver<RenderThreadInputMessage>,
    output_channel: mpsc::Sender<RenderThreadOutputMessage>,
    initial_buffer: *mut Buffer,
    thread_id: usize,
) {
    let mut buffer: &mut Buffer = unsafe {initial_buffer.as_mut().unwrap()};
    let mut last_render_micros: u128 = 0;
    for message in input_channel.iter() {
        match message {
            RenderThreadInputMessage::Stop => break,
            RenderThreadInputMessage::NewBuffer(new_buffer) => buffer = new_buffer,
            RenderThreadInputMessage::RenderInfo => {
                output_channel.send(RenderThreadOutputMessage::RenderInfo(
                    RenderInfo {
                        thread_id,
                        render_micros: last_render_micros,
                    }
                )).unwrap();
            },
            RenderThreadInputMessage::RenderLines(y0, y1, commands) => {
                let frame_start = std::time::Instant::now();
                let x0 = 0f32;
                let x1 = buffer.width as f32;
                let rect = Rect{x0, x1, y0: y0 as f32, y1: y1 as f32};

                for command in commands.iter() {
                    match command {
                        RenderCommand::RenderTri(tri_info) => {
                            buffer.render_tri(
                                (tri_info.a, tri_info.b, tri_info.c),
                                tri_info.depth_test_enabled,
                                tri_info.input_clip_rect
                                    .map(|command_rect| command_rect.intersect(&rect))
                                    .or(Some(rect)),
                                match &tri_info.texture {
                                    Some(arc_t) => Some(arc_t),
                                    None => None,
                                },
                            );
                        },
                        RenderCommand::Clear(color) => {
                            for d in buffer.depth[y0*buffer.width .. y1*buffer.width].iter_mut() {
                                *d = f32::INFINITY;
                            }
                            for c in buffer.color[y0*buffer.width .. y1*buffer.width].iter_mut() {
                                *c = *color;
                            }
                        },
                    }
                }
                output_channel.send(RenderThreadOutputMessage::RenderDone(thread_id)).unwrap();
                last_render_micros = frame_start.elapsed().as_micros();
            },
        }
    }
}

struct RenderTriCommand {
    a: Vert,
    b: Vert,
    c: Vert,
    depth_test_enabled: bool,
    input_clip_rect: Option<Rect>,
    texture: Option<Arc<Texture>>,
}

enum RenderCommand {
    RenderTri(RenderTriCommand),
    Clear(u32),
}

enum RenderThreadInputMessage {
    NewBuffer(&'static mut Buffer),
    RenderLines(usize, usize, Arc<Vec<RenderCommand>>),
    RenderInfo,
    Stop,
}

enum RenderThreadOutputMessage {
    RenderDone(usize),
    RenderInfo(RenderInfo),
}

struct RenderInfo {
    thread_id: usize,
    render_micros: u128,
}

enum TextureFilter {
    Nearest,
    Linear,
}

struct Texture {
    // Color as argb
    color: Vec<u32>,
    filtering: TextureFilter,
    width: usize,
    height: usize,
}

impl Texture {
    fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Self {
        assert_eq!(width*height*4, rgba.len());
        let mut color: Vec<u32> = vec![0; width * height];
        for y in 0..height {
            for x in 0..width {

                let pixel =
                    // R
                      (rgba[(x + y*width)*4 + 0] as u32) << 16
                    // G
                    | (rgba[(x + y*width)*4 + 1] as u32) << 8
                    // B
                    | (rgba[(x + y*width)*4 + 2] as u32)
                    // A
                    | (rgba[(x + y*width)*4 + 3] as u32) << 24;

                color[x + y*width] = pixel;
            }
        }

        let filtering = TextureFilter::Nearest;
        Self {color, filtering, width, height}
    }

    fn interp(&self, uv: Vec2) -> u32 {
        let mut uv = uv;
        uv.x = uv.x * self.width as f32;
        uv.y = uv.y * self.height as f32;

        let u = (uv.x.max(0f32) as usize).min(self.width - 1);
        let v = (uv.y.max(0f32) as usize).min(self.height - 1);

        match self.filtering {
            TextureFilter::Nearest => {
                self.color[u + v*self.width]
            },
            TextureFilter::Linear => unimplemented!(),
        }
    }
}

struct Buffer {
    // Color as Xrgb
    color: Vec<u32>,
    depth: Vec<f32>,
    width: usize,
    height: usize,
}

impl Buffer {
    fn new(width: usize, height: usize) -> Self {
        let color: Vec<u32> = vec![0; width * height];
        let depth: Vec<f32> = vec![f32::INFINITY; width * height];

        Self {color, depth, width, height}
    }

    fn clear(&mut self, color: u32) {
        // Clear buffers. At 1080p clearing these buffers takes 1.0 ms.
        for d in self.depth.iter_mut() {
            *d = f32::INFINITY
        }

        for c in self.color.iter_mut() {
            *c = color
        }
    }

    fn render_tri(
        &mut self,
        (a,b,c): (Vert, Vert, Vert),
        depth_test_enabled: bool,
        input_clip_rect: Option<Rect>,
        texture: Option<&Texture>,
    ) {
        let screen_rect = Rect::new(0f32, WIDTH_F, 0f32, HEIGHT_F);
        let clip_rect = input_clip_rect
            .map(|r| r.intersect(&screen_rect))
            .unwrap_or(screen_rect);

        //// Project the triangle onto the screen
        //let a_s = Vec3::new(a.x/a.z*HEIGHT_F + WIDTH_F/2.0f32, -a.y/a.z*HEIGHT_F + HEIGHT_F/2.0f32, a.z);
        //let b_s = Vec3::new(b.x/b.z*HEIGHT_F + WIDTH_F/2.0f32, -b.y/b.z*HEIGHT_F + HEIGHT_F/2.0f32, b.z);
        //let c_s = Vec3::new(c.x/c.z*HEIGHT_F + WIDTH_F/2.0f32, -c.y/c.z*HEIGHT_F + HEIGHT_F/2.0f32, c.z);
        let a_s = a.pos;
        let b_s = b.pos;
        let c_s = c.pos;

        // Divide vertex attribute by vertex z for perspective correction
        let a_c = a.col / a_s.z;
        let b_c = b.col / b_s.z;
        let c_c = c.col / c_s.z;

        let a_uv = a.uv / a_s.z;
        let b_uv = b.uv / b_s.z;
        let c_uv = c.uv / c_s.z;

        let inv_triangle_area = ((b_s.y - c_s.y)*(a_s.x - c_s.x) + (c_s.x - b_s.x)*(a_s.y - c_s.y)).recip();

        // Max and min y that this triangle touches. The max is inclusive in
        // this range.
        //
        // Min of all the vertices, ensuring we don't go below 0
        let min_y = a_s.y.min(b_s.y).min(c_s.y).round().max(clip_rect.y0);

        // Max of all the vertices, ensuring we don't go beyond the screen extents
        // We add one so I don't need to figure out the proper math here,
        // if y goes beyond where the triangle actually is, the min_x and
        // max_x will be set to values that cause no pixels to be rendered,
        // which is what we want. It means we need to calculate min_x and
        // max_x an extra time for some triangles, but that's not a big deal.
        let max_y = a_s.y.max(b_s.y).max(c_s.y).round().add(1f32).min(clip_rect.y1).max(0f32);

        assert!(min_y >= 0f32);
        assert!(max_y >= 0f32);

        let min_y = min_y as usize;
        let max_y = max_y as usize;

        let lines = [
            LineSegment::new(a_s.xy(),b_s.xy()),
            LineSegment::new(a_s.xy(),c_s.xy()),
            LineSegment::new(b_s.xy(),c_s.xy()),
        ];

        for y in min_y..max_y {
            let db_actual_y = y;
            // At the given y, what's the leftmost x pixel

            // Limiting the x range reduces the time to render (the triangle
            // I was testing with) from 2.0ms to 1.0ms. Which is pretty substantial!
            let min_x = lines[0].x_for_y(y as f32).unwrap_or(WIDTH_F)
                .min(lines[1].x_for_y(y as f32).unwrap_or(WIDTH_F))
                .min(lines[2].x_for_y(y as f32).unwrap_or(WIDTH_F))
                .round()
                .max(clip_rect.x0);

            let max_x = lines[0].x_for_y(y as f32).unwrap_or(0f32)
                .max(lines[1].x_for_y(y as f32).unwrap_or(0f32))
                .max(lines[2].x_for_y(y as f32).unwrap_or(0f32))
                .round()
                .min(clip_rect.x1)
                .max(0f32);

            assert!(min_x >= 0f32);
            assert!(max_x >= 0f32);

            let min_x = min_x as usize;
            let max_x = max_x as usize;

            for x in min_x..max_x {
                let p = Vec2::new(x as f32, y as f32);

                let bary_a_area = (b_s.y - c_s.y)*(p.x - c_s.x) + (c_s.x - b_s.x)*(p.y - c_s.y);
                let bary_b_area = (c_s.y - a_s.y)*(p.x - c_s.x) + (a_s.x - c_s.x)*(p.y - c_s.y);

                let bary_a = bary_a_area * inv_triangle_area;
                let bary_b = bary_b_area * inv_triangle_area;
                let bary_c = 1f32 - bary_a - bary_b;

                let (c, d) = {
                    // Interpolate the z values
                    let z: f32 = (bary_a*a_s.z.recip() + bary_b*b_s.z.recip() + bary_c*c_s.z.recip()).recip();

                    // This is the color as determined by the vertex attribute
                    let (vr, vg, vb) = {
                        let r: u32 = (((bary_a*a_c.x + bary_b*b_c.x + bary_c*c_c.x)*z*255f32) as u32) & 0xff;
                        let g: u32 = (((bary_a*a_c.y + bary_b*b_c.y + bary_c*c_c.y)*z*255f32) as u32) & 0xff;
                        let b: u32 = (((bary_a*a_c.z + bary_b*b_c.z + bary_c*c_c.z)*z*255f32) as u32) & 0xff;

                        (r, g, b)
                    };

                    let c = if let Some(texture) = texture {
                        let u: f32 = (bary_a*a_uv.x + bary_b*b_uv.x + bary_c*c_uv.x)*z;
                        let v: f32 = (bary_a*a_uv.y + bary_b*b_uv.y + bary_c*c_uv.y)*z;

                        if 0.25f32 < u && u < 0.75f32 {
                            let mbp = "foo";
                        }

                        let p = texture.interp(Vec2::new(u, v));

                        // This is the color as determined by the texture
                        let ta = (p & 0xff00_0000) >> 24;
                        let tr = (p & 0x00ff_0000) >> 16;
                        let tg = (p & 0x0000_ff00) >> 8;
                        let tb = (p & 0x0000_00ff);

                        // For now, we're not actually blending with the color
                        // underneath

                        // Blend the alpha, texture color, and vertex color for each channel
                        let r = (tr*vr) >> 8;
                        let g = (tg*vg) >> 8;
                        let b = (tb*vb) >> 8;

                        ta << 24 | r << 16 | g << 8 | b
                    } else {
                        0xff00_0000 | vr << 16 | vg << 8 | vb
                    };

                    (c, z)
                };

                if d <= self.depth[x+y*self.width] || !depth_test_enabled {
                    let a = (c & 0xff00_0000) >> 24;

                    let alpha_mask = a << 16 | a << 8 | a;
                    let inv_alpha_mask = alpha_mask ^ 0xffff_ffff;

                    let current_color = self.color[x+y*self.width];
                    self.color[x+y*self.width] = c & alpha_mask | current_color & inv_alpha_mask;
                    self.depth[x+y*self.width] = d;
                }
            }
        }
    }
}

fn core_count() -> usize {
    let mut returned_length: u32 = 0;

    // Determine how large of a buffer to allocate for the processor information, in bytes
    // We assume this returns a result of ERROR_INSUFFICIENT_BUFFER
    unsafe { winapi::um::sysinfoapi::GetLogicalProcessorInformation(ptr::null_mut(), &mut returned_length) };

    let proc_info_count = returned_length as usize / mem::size_of::<winapi::um::winnt::SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
    let mut proc_info_buffer: Vec<winapi::um::winnt::SYSTEM_LOGICAL_PROCESSOR_INFORMATION> = Vec::with_capacity(proc_info_count);
    unsafe { proc_info_buffer.set_len(proc_info_count) };

    // Returns 0 on failure
    let success = unsafe { winapi::um::sysinfoapi::GetLogicalProcessorInformation(proc_info_buffer.as_mut_ptr(), &mut returned_length) };
    if success == 0 {
        panic!("Could not get processor information");
    }

    proc_info_buffer.iter()
        .filter(|proc| proc.Relationship == 0)
        .count()
}

fn main() {
    let mut buffer = Buffer::new(WIDTH_USIZE, HEIGHT_USIZE);

    let mut window = Window::new(
        "SHM Renderer",
        WIDTH as usize,
        HEIGHT as usize,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    let mut ctx = imgui::Context::create();
    // We don't seem to get draw commands the first frame if we read from the ini
    ctx.set_ini_filename(None);

    ctx.io_mut().display_size[0] = WIDTH_F;
    ctx.io_mut().display_size[1] = HEIGHT_F;
    ctx.io_mut().font_global_scale = 1.0;

    ctx.fonts().add_font(&[
        FontSource::DefaultFontData { config: Some(FontConfig::default()) },
    ]);

    let texture: Arc<Texture> = Arc::new({
        let mut fonts = ctx.fonts();
        let font = fonts.build_rgba32_texture();

        Texture::from_rgba(font.width as usize, font.height as usize, font.data)
    });

    println!("Font size: {}x{}", texture.width, texture.height);

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut t: f32 = 3.5f32;
    let mut frame_number = 0;

    let mut triangle_width: f32 = 61.04512;
    let mut triangle_height: f32 = 33.333336;

    let mut last_command_count = 0;

    let imgui_enabled = true;
    let checkerboard = true;

    let mut frame_times_ms: Vec<f32> = vec![16.6; 60];
    let mut thread_frame_times_ms: [Vec<f32>; 8] = [
        vec![16.6; 60],
        vec![16.6; 60],
        vec![16.6; 60],
        vec![16.6; 60],
        vec![16.6; 60],
        vec![16.6; 60],
        vec![16.6; 60],
        vec![16.6; 60],
    ];

    let mut micros_to_build_commands: u128 = 0;

    let physical_proc_count = core_count();
    println!("Cores: {}", physical_proc_count);

    let (command_tx0, command_rx0) = mpsc::channel();
    let (command_tx1, command_rx1) = mpsc::channel();
    let (command_tx2, command_rx2) = mpsc::channel();
    let (command_tx3, command_rx3) = mpsc::channel();
    let (command_tx4, command_rx4) = mpsc::channel();
    let (command_tx5, command_rx5) = mpsc::channel();
    let (command_tx6, command_rx6) = mpsc::channel();
    let (command_tx7, command_rx7) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();

    let buffer_ptr: *mut Buffer = &mut buffer;

    let ptr1 = buffer_ptr as usize;
    let ptr2 = buffer_ptr as usize;

    let result_tx0 = result_tx.clone();
    let result_tx1 = result_tx.clone();
    let result_tx2 = result_tx.clone();
    let result_tx3 = result_tx.clone();
    let result_tx4 = result_tx.clone();
    let result_tx5 = result_tx.clone();
    let result_tx6 = result_tx.clone();
    let result_tx7 = result_tx.clone();
    thread::spawn(move || run_render_thread(
        command_rx0,
        result_tx0,
        ptr1 as *mut Buffer,
        0,
    ));
    thread::spawn(move || run_render_thread(
        command_rx1,
        result_tx1,
        ptr1 as *mut Buffer,
        1,
    ));
    thread::spawn(move || run_render_thread(
        command_rx2,
        result_tx2,
        ptr1 as *mut Buffer,
        2,
    ));
    thread::spawn(move || run_render_thread(
        command_rx3,
        result_tx3,
        ptr1 as *mut Buffer,
        3,
    ));
    thread::spawn(move || run_render_thread(
        command_rx4,
        result_tx4,
        ptr1 as *mut Buffer,
        4,
    ));
    thread::spawn(move || run_render_thread(
        command_rx5,
        result_tx5,
        ptr1 as *mut Buffer,
        5,
    ));
    thread::spawn(move || run_render_thread(
        command_rx6,
        result_tx6,
        ptr1 as *mut Buffer,
        6,
    ));
    thread::spawn(move || run_render_thread(
        command_rx7,
        result_tx7,
        ptr1 as *mut Buffer,
        7,
    ));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_start = std::time::Instant::now();

        let mut render_commands = vec![RenderCommand::Clear(0x00ff_00ff)];

        let mouse_pos = window.get_mouse_pos(minifb::MouseMode::Clamp).unwrap_or((0.0, 0.0));
        ctx.io_mut().mouse_pos[0] = mouse_pos.0;
        ctx.io_mut().mouse_pos[1] = mouse_pos.1;

        ctx.io_mut().mouse_down[0] = window.get_mouse_down(minifb::MouseButton::Left);

        let ui = ctx.frame();

        imgui::Window::new(im_str!("Hello World!"))
            .size([400.0, 600.0], Condition::FirstUseEver)
            .build(&ui, || {
                let total_frame_time = frame_times_ms.iter().sum::<f32>() / frame_times_ms.len() as f32;
                ui.text(format!(
                    "Build Time (us): {:>6}",
                    micros_to_build_commands
                ));
                ui.text(format!(
                    "Frame Time (ms): {:.2}",
                    total_frame_time
                ));
                ui.text(format!("Thread 0 (ms): {:.2}", thread_frame_times_ms[0].iter().sum::<f32>() / 60.0));
                ui.text(format!("Thread 1 (ms): {:.2}", thread_frame_times_ms[1].iter().sum::<f32>() / 60.0));
                ui.text(format!("Thread 2 (ms): {:.2}", thread_frame_times_ms[2].iter().sum::<f32>() / 60.0));
                ui.text(format!("Thread 3 (ms): {:.2}", thread_frame_times_ms[3].iter().sum::<f32>() / 60.0));
                ui.text(format!("Thread 4 (ms): {:.2}", thread_frame_times_ms[4].iter().sum::<f32>() / 60.0));
                ui.text(format!("Thread 5 (ms): {:.2}", thread_frame_times_ms[5].iter().sum::<f32>() / 60.0));
                ui.text(format!("Thread 6 (ms): {:.2}", thread_frame_times_ms[6].iter().sum::<f32>() / 60.0));
                ui.text(format!("Thread 7 (ms): {:.2}", thread_frame_times_ms[7].iter().sum::<f32>() / 60.0));
                ui.text(format!(
                    "Command Count: {}",
                    last_command_count
                ));

                {
                    let max_frame_time = 16.6;
                    let background_bar_width = ui.calc_item_width();
                    let bar_width = total_frame_time / max_frame_time * ui.calc_item_width();

                    let size: [f32; 2] = [bar_width, ui.frame_height()];
                    let bg_size: [f32; 2] = [background_bar_width, ui.frame_height()];
                    let draw_list = ui.get_window_draw_list();

                    let p0: [f32; 2] = ui.cursor_screen_pos();
                    let p1: [f32; 2] = [p0[0] + f32::min(bar_width, background_bar_width), p0[1] + size[1]];

                    let bg0: [f32; 2] = ui.cursor_screen_pos();
                    let bg1: [f32; 2] = [p0[0] + background_bar_width, p0[1] + size[1]];

                    let col_bg = 0xff33_3333;
                    let col_a = 0xff00_ff00;

                    draw_list.add_rect_filled_multicolor(bg0, bg1, col_bg, col_bg, col_bg, col_bg);
                    draw_list.add_rect_filled_multicolor(p0, p1, col_a, col_a, col_a, col_a);

                    if bar_width > background_bar_width {
                        draw_list.add_rect_filled_multicolor(
                            [p0[0] + background_bar_width, p0[1]],
                            [p0[0] + bar_width, p0[1] + size[1]],
                            0xff00_00ff,
                            0xff00_00ff,
                            0xff00_00ff,
                            0xff00_00ff,
                        );
                    }


                    ui.invisible_button(im_str!("##gradient1"), size);
                }

                imgui::PlotLines::new(&ui, im_str!(""), &frame_times_ms)
                    .graph_size([400.0, 80.0])
                    .scale_min(0.0)
                    .scale_max(16.6)
                    .values_offset(frame_number % 60)
                    .overlay_text(
                        &im_str!(
                            "Frame Time: {:.2}",
                            frame_times_ms.iter().sum::<f32>() / 60.0
                         )
                     )
                    .build();
                for thread_id in 0 .. 8 {
                    imgui::PlotLines::new(&ui, im_str!(""), &thread_frame_times_ms[thread_id])
                        .graph_size([150.0, 50.0])
                        .scale_min(0.0)
                        .scale_max(16.6)
                        .values_offset(frame_number % 60)
                        .overlay_text(
                            &im_str!(
                                "Thread {}: {:.2}",
                                thread_id,
                                thread_frame_times_ms[thread_id].iter().sum::<f32>() / 60.0
                             )
                         )
                        .build();
                }

                ui.separator();

                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));

                ui.separator();

                imgui::Slider::new(im_str!("triangle_width"), 5.0 ..= 50.0)
                    .build(&ui, &mut triangle_width);
                imgui::Slider::new(im_str!("triangle_height"), 5.0 ..= 50.0)
                    .build(&ui, &mut triangle_height);
            });

        let draw_data = ui.render();
        if checkerboard {
            for x in 0 .. 50 {
                for y in 0 .. 50 {
                    let col = match (x+y) % 10 {
                        0 => Vec3::new(1.0, 0.0, 0.0),
                        1 => Vec3::new(0.0, 1.0, 0.0),
                        2 => Vec3::new(0.0, 0.0, 1.0),
                        3 => Vec3::new(1.0, 1.0, 0.0),
                        4 => Vec3::new(1.0, 0.0, 1.0),
                        5 => Vec3::new(0.0, 1.0, 1.0),
                        6 => Vec3::new(0.3, 0.3, 0.3),
                        7 => Vec3::new(0.7, 0.7, 0.7),
                        8 => Vec3::new(1.0, 1.0, 1.0),
                        _ => Vec3::new(0.5, 1.0, 0.5),
                    };

                    render_commands.push(
                        RenderCommand::RenderTri(
                            RenderTriCommand {
                                a: Vert::newf32(Vec4::new(0.0 + x as f32 *triangle_width,     y as f32 * triangle_height,     1.0, 1.0), col, Vec2::new(0f32, 0f32)),
                                b: Vert::newf32(Vec4::new(0.0 + x as f32 *triangle_width,     (y+1) as f32 * triangle_height, 1.0, 1.0), col, Vec2::new(0f32, 0f32)),
                                c: Vert::newf32(Vec4::new(0.0 + (x+1) as f32 *triangle_width, (y+1) as f32 * triangle_height, 1.0, 1.0), col, Vec2::new(0f32, 0f32)),
                                depth_test_enabled: false,
                                input_clip_rect: None,
                                texture: None,
                            }
                        )
                    );
                    render_commands.push(
                        RenderCommand::RenderTri(
                            RenderTriCommand {
                                a: Vert::newf32(Vec4::new(0.0 + (x+1) as f32 *triangle_width, (y+1) as f32 * triangle_height, 1.0, 1.0), col, Vec2::new(0f32, 0f32)),
                                b: Vert::newf32(Vec4::new(0.0 + (x+1) as f32 *triangle_width, y as f32 * triangle_height,     1.0, 1.0), col, Vec2::new(0f32, 0f32)),
                                c: Vert::newf32(Vec4::new(0.0 + x as f32 *triangle_width,     y as f32 * triangle_height,     1.0, 1.0), col, Vec2::new(0f32, 0f32)),
                                depth_test_enabled: false,
                                input_clip_rect: None,
                                texture: None,
                            }
                        )
                    );
                }
            }
        }

        for draw_list in draw_data.draw_lists() {
            let idx_buffer = draw_list.idx_buffer();
            let vtx_buffer = draw_list.vtx_buffer();

            for draw_cmd in draw_list.commands() {
                match draw_cmd {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        let idx_offset = cmd_params.idx_offset;
                        let vtx_offset = cmd_params.vtx_offset; // Always 0...?

                        let x0 = cmd_params.clip_rect[0];
                        let y0 = cmd_params.clip_rect[1];
                        let x1 = cmd_params.clip_rect[2];
                        let y1 = cmd_params.clip_rect[3];

                        let clip_rect = Rect {x0, x1, y0, y1};
                        for index in (idx_offset..(idx_offset+count)).step_by(3) {
                            let v0 = vtx_buffer[idx_buffer[index + 0] as usize];
                            let v1 = vtx_buffer[idx_buffer[index + 1] as usize];
                            let v2 = vtx_buffer[idx_buffer[index + 2] as usize];

                            if imgui_enabled {
                                render_commands.push(
                                    RenderCommand::RenderTri(
                                        RenderTriCommand {
                                            a: Vert::new_from_imgui(Vec4::new(v0.pos[0], v0.pos[1], 1.0, 1.0), v0.col, v0.uv),
                                            b: Vert::new_from_imgui(Vec4::new(v1.pos[0], v1.pos[1], 1.0, 1.0), v1.col, v1.uv),
                                            c: Vert::new_from_imgui(Vec4::new(v2.pos[0], v2.pos[1], 1.0, 1.0), v2.col, v2.uv),
                                            depth_test_enabled: false,
                                            input_clip_rect: Some(clip_rect),
                                            texture: Some(texture.clone()),
                                        }
                                    )
                                );
                            }
                        }
                    },
                    imgui::DrawCmd::ResetRenderState => unimplemented!("ResetRenderState"),
                    imgui::DrawCmd::RawCallback { callback, raw_cmd } => unimplemented!("RawCallback"),
                }
            }
        }

        let mut input_state: winapi::um::xinput::XINPUT_STATE = unsafe { mem::zeroed() };
        unsafe { winapi::um::xinput::XInputGetState(0, &mut input_state) };

        // Thumbstick positions
        let tx: f32 = input_state.Gamepad.sThumbLX as f32;
        let _ty: f32 = input_state.Gamepad.sThumbLY as f32;


        micros_to_build_commands = frame_start.elapsed().as_micros();
        last_command_count = render_commands.len();

        {
            let arc_commands: Arc<Vec<RenderCommand>> = Arc::new(render_commands);
            command_tx0.send(
                RenderThreadInputMessage::RenderLines(0, 135, arc_commands.clone())
            ).unwrap();
            command_tx1.send(
                RenderThreadInputMessage::RenderLines(135, 270, arc_commands.clone())
            ).unwrap();
            command_tx2.send(
                RenderThreadInputMessage::RenderLines(270, 405, arc_commands.clone())
            ).unwrap();
            command_tx3.send(
                RenderThreadInputMessage::RenderLines(405, 540, arc_commands.clone())
            ).unwrap();
            command_tx4.send(
                RenderThreadInputMessage::RenderLines(540, 675, arc_commands.clone())
            ).unwrap();
            command_tx5.send(
                RenderThreadInputMessage::RenderLines(675, 810, arc_commands.clone())
            ).unwrap();
            command_tx6.send(
                RenderThreadInputMessage::RenderLines(810, 945, arc_commands.clone())
            ).unwrap();
            command_tx7.send(
                RenderThreadInputMessage::RenderLines(945, 1080, arc_commands.clone())
            ).unwrap();

            result_rx.recv().unwrap();
            result_rx.recv().unwrap();
            result_rx.recv().unwrap();
            result_rx.recv().unwrap();
            result_rx.recv().unwrap();
            result_rx.recv().unwrap();
            result_rx.recv().unwrap();
            result_rx.recv().unwrap();


            command_tx0.send(RenderThreadInputMessage::RenderInfo).unwrap();
            command_tx1.send(RenderThreadInputMessage::RenderInfo).unwrap();
            command_tx2.send(RenderThreadInputMessage::RenderInfo).unwrap();
            command_tx3.send(RenderThreadInputMessage::RenderInfo).unwrap();
            command_tx4.send(RenderThreadInputMessage::RenderInfo).unwrap();
            command_tx5.send(RenderThreadInputMessage::RenderInfo).unwrap();
            command_tx6.send(RenderThreadInputMessage::RenderInfo).unwrap();
            command_tx7.send(RenderThreadInputMessage::RenderInfo).unwrap();
        }



        // END FRAME
        // Finished rendering frame

        let current_frame_time_ms = frame_start.elapsed().as_secs_f32() * 1000.0;

        let frame_time_idx = frame_number % frame_times_ms.len();
        frame_times_ms[frame_time_idx] = current_frame_time_ms;

        let thread_frame_times = [
            result_rx.recv().unwrap(),
            result_rx.recv().unwrap(),
            result_rx.recv().unwrap(),
            result_rx.recv().unwrap(),
            result_rx.recv().unwrap(),
            result_rx.recv().unwrap(),
            result_rx.recv().unwrap(),
            result_rx.recv().unwrap(),
        ];

        for msg in thread_frame_times.iter() {
            match msg {
                RenderThreadOutputMessage::RenderInfo(info) => {
                    thread_frame_times_ms[info.thread_id][frame_time_idx] = info.render_micros as f32 / 1000.0
                },
                _ => panic!("Got unexpected message from render thread")
            }
        }

        // Multiply by 100 and clamp betwwn 0 and 255
        //let scaled_z = (z*50f32).min(255f32).max(0f32) as u32;

        // Switch to the depth buffer when A is pressed
        if (input_state.Gamepad.wButtons & XINPUT_GAMEPAD_A) != 0 {
            let int_depth_buffer: Vec<u32> = buffer.depth.iter().map(|f| {
                let channel = f.mul(50f32).min(255f32).max(0f32) as u32;
                channel << 16 | channel << 8 | channel
            }).collect();
            window
                .update_with_buffer(&int_depth_buffer, buffer.width, buffer.height)
                .unwrap();
        } else {
            window
                .update_with_buffer(&buffer.color, buffer.width, buffer.height)
                .unwrap();
        }

        if 8_000f32 < tx.abs() {
            t += tx / 1_000_000f32;
        }
        frame_number += 1;
    }
}

