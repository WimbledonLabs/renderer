#![allow(dead_code, unused_imports, unused_variables)]

use std::mem;
use std::ops::{Add, Mul, Sub};
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

fn point_in_tri(p: Vec2, t: (Vec2, Vec2, Vec2)) -> bool {
    let s0 = (p.x - t.0.x)*(t.1.y - t.0.y) - (p.y - t.0.y)*(t.1.x - t.0.x) < 0f32;
    let s1 = (p.x - t.1.x)*(t.2.y - t.1.y) - (p.y - t.1.y)*(t.2.x - t.1.x) < 0f32;
    let s2 = (p.x - t.2.x)*(t.0.y - t.2.y) - (p.y - t.2.y)*(t.0.x - t.2.x) < 0f32;

    s0 == s1 && s1 == s2
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

struct Vert {
    pos: Vec4,
    col: Vec3,
}

impl Vert {
    fn new(pos: Vec4, col: [u8; 4]) -> Self {
        let r = col[0] as f32 / 255f32;
        let g = col[1] as f32 / 255f32;
        let b = col[2] as f32 / 255f32;

        let col = Vec3::new(r,g,b);
        Self {pos, col}
    }

    fn newf32(pos: Vec4, col: Vec3) -> Self {
        Self {pos, col}
    }

}

fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH as usize * HEIGHT as usize];
    let mut depth_buffer: Vec<f32> = vec![f32::INFINITY; WIDTH as usize * HEIGHT as usize];

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

    ctx.fonts().build_rgba32_texture();


    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    let mut t: f32 = 3.5f32;
    let mut frame_number = 0;
    let mut frame_time_accumulator: f32 = 0f32;

    let mut triangle_width: f32 = 61.04512;
    let mut triangle_height: f32 = 33.333336;

    let depth_test_enabled = false;
    let imgui_enabled = true;

    let mut triangle_count: u32 = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let frame_start = SystemTime::now();

        let mouse_pos = window.get_mouse_pos(minifb::MouseMode::Clamp).unwrap_or((0.0, 0.0));
        ctx.io_mut().mouse_pos[0] = mouse_pos.0;
        ctx.io_mut().mouse_pos[1] = mouse_pos.1;

        ctx.io_mut().mouse_down[0] = window.get_mouse_down(minifb::MouseButton::Left);

        let ui = ctx.frame();

        imgui::Window::new(im_str!("Hello World!"))
            .size([400.0, 600.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(im_str!("Hello world!"));
                imgui::Slider::new(im_str!("count"), 0 ..= 60000)
                    .build(&ui, &mut triangle_count);
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("Hello world!"));
                imgui::Slider::new(im_str!("triangle_width"), 0.1 ..= 200.0)
                    .build(&ui, &mut triangle_width);
                imgui::Slider::new(im_str!("triangle_height"), 0.0 ..= 200.0)
                    .build(&ui, &mut triangle_height);
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.text(im_str!("Hello world!"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });

        let draw_data = ui.render();
        let mut tris: Vec<(Vert, Vert, Vert)> = Vec::with_capacity(draw_data.total_idx_count as usize + 20);

        for x in (0 .. 50) {
            for y in (0 .. 50) {
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
                tris.push(
                    (
                    Vert::newf32(Vec4::new(0.0 + x as f32 *triangle_width,     y as f32 * triangle_height,          1.0, 1.0), col),
                    Vert::newf32(Vec4::new(0.0 + x as f32 *triangle_width,     (y+1) as f32 * triangle_height, 1.0, 1.0), col),
                    Vert::newf32(Vec4::new(0.0 + (x+1) as f32 *triangle_width, (y+1) as f32 * triangle_height, 1.0, 1.0), col),
                    )
                );
                tris.push(
                    (
                    Vert::newf32(Vec4::new(0.0 + (x+1) as f32 *triangle_width, (y+1) as f32 * triangle_height, 1.0, 1.0), col),
                    Vert::newf32(Vec4::new(0.0 + (x+1) as f32 *triangle_width, y as f32 * triangle_height,          1.0, 1.0), col),
                    Vert::newf32(Vec4::new(0.0 + x as f32 *triangle_width,     y as f32 * triangle_height,          1.0, 1.0), col),
                    )
                );
            }
        }


        for draw_list in draw_data.draw_lists() {
            let idx_buffer = draw_list.idx_buffer();
            let vtx_buffer = draw_list.vtx_buffer();

            for draw_cmd in draw_list.commands() {
                match draw_cmd {
                    imgui::DrawCmd::Elements { count, cmd_params } => {
                        let idx_offset = cmd_params.idx_offset;
                        for index in (idx_offset..(idx_offset+count)).step_by(3) {
                            let v0 = vtx_buffer[idx_buffer[index + 0] as usize];
                            let v1 = vtx_buffer[idx_buffer[index + 1] as usize];
                            let v2 = vtx_buffer[idx_buffer[index + 2] as usize];

                            if imgui_enabled {
                                tris.push((
                                    Vert::new(Vec4::new(v0.pos[0], v0.pos[1], 1.0, 1.0), v0.col),
                                    Vert::new(Vec4::new(v1.pos[0], v1.pos[1], 1.0, 1.0), v1.col),
                                    Vert::new(Vec4::new(v2.pos[0], v2.pos[1], 1.0, 1.0), v2.col),
                                ));
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

        // 3D isosceles triangle with its center at the origin, y-up
        let a_blarg = Vec4::new(-1.0f32,  1.0f32, 0.0f32, 1.0f32);
        let b_blarg = Vec4::new(-1.0f32, -1.0f32, 0.0f32, 1.0f32);
        let c_blarg = Vec4::new( 1.0f32,  1.0f32, 0.0f32, 1.0f32);

        fn y_axis_rotate(p: Vec4, theta: f32) -> Vec4 {
            Vec4::new(
                p.x*theta.cos() - p.z*theta.sin(),
                p.y,
                p.x*theta.sin() + p.z*theta.cos(),
                1.0f32,
            )
        }

        // Rotate the triangle, and push it back 4 units
        let a = y_axis_rotate(a_blarg, t) + Vec4::new(0.0f32, 0.0f32, 4.0f32, 0.0f32);
        let b = y_axis_rotate(b_blarg, t) + Vec4::new(0.0f32, 0.0f32, 4.0f32, 0.0f32);
        let c = y_axis_rotate(c_blarg, t) + Vec4::new(0.0f32, 0.0f32, 4.0f32, 0.0f32);

        // Rotate the triangle more, and push it back 4 units
        let d = y_axis_rotate(a_blarg, 2f32*t) + Vec4::new(0.0f32, 0.0f32, 4.0f32, 0.0f32);
        let e = y_axis_rotate(b_blarg, 2f32*t) + Vec4::new(0.0f32, 0.0f32, 4.0f32, 0.0f32);
        let f = y_axis_rotate(c_blarg, 2f32*t) + Vec4::new(0.0f32, 0.0f32, 4.0f32, 0.0f32);

        // Clear buffers. At 1080p clearing these buffers takes 1.0 ms.
        for d in depth_buffer.iter_mut() {
            *d = f32::INFINITY
        }

        for c in buffer.iter_mut() {
            *c = CLEAR_COLOR
        }

        triangle_count = tris.len() as u32;

        for (a,b,c) in tris.iter() {
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

            let _a_uv = Vec2::new(0.0f32, 1.0f32) / a_s.z;
            let _b_uv = Vec2::new(0.0f32, 0.0f32) / b_s.z;
            let _c_uv = Vec2::new(1.0f32, 1.0f32) / c_s.z;

            let inv_triangle_area = ((b_s.y - c_s.y)*(a_s.x - c_s.x) + (c_s.x - b_s.x)*(a_s.y - c_s.y)).recip();

            // Max and min y that this triangle touches. The max is inclusive in
            // this range.
            //
            // Min of all the vertices, ensuring we don't go below 0
            let min_y = (a_s.y.min(b_s.y).min(c_s.y).round() as usize).saturating_sub(0).max(0);

            // Max of all the vertices, ensuring we don't go beyond the screen extents
            // We add one so I don't need to figure out the proper math here,
            // if y goes beyond where the triangle actually is, the min_x and
            // max_x will be set to values that cause no pixels to be rendered,
            // which is what we want. It means we need to calculate min_x and
            // max_x an extra time for some triangles, but that's not a big deal.
            let max_y = ((a_s.y.max(b_s.y).max(c_s.y).round() as usize) + 1).min(HEIGHT_USIZE);

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
                    .round() as usize;

                let max_x = lines[0].x_for_y(y as f32).unwrap_or(0f32)
                    .max(lines[1].x_for_y(y as f32).unwrap_or(0f32))
                    .max(lines[2].x_for_y(y as f32).unwrap_or(0f32))
                    .round() as usize;

                if min_x < 10 || max_x > 900 {
                    let _bp = 1;
                }

                let min_x = min_x.max(0);
                let max_x = max_x.min(WIDTH_USIZE);

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

                        let cr: u32 = (((bary_a*a_c.x + bary_b*b_c.x + bary_c*c_c.x)*z*255f32) as u32) & 0xff;
                        let cg: u32 = (((bary_a*a_c.y + bary_b*b_c.y + bary_c*c_c.y)*z*255f32) as u32) & 0xff;
                        let cb: u32 = (((bary_a*a_c.z + bary_b*b_c.z + bary_c*c_c.z)*z*255f32) as u32) & 0xff;

                        (cr << 16 | cg << 8 | cb, z)
                    };

                    if d <= depth_buffer[x+y*WIDTH_USIZE] || !depth_test_enabled {
                        buffer[x+y*WIDTH_USIZE] = c;
                        depth_buffer[x+y*WIDTH_USIZE] = d;
                    }
                }
            }
        }

        let frame_time = SystemTime::now().duration_since(frame_start).unwrap();
        frame_time_accumulator += frame_time.as_secs_f32();

        // Every 60 frame, print the seconds per frame
        if frame_number % 60 == 0 {
            println!("frame_time:     {:?}", frame_time_accumulator / 60f32);
            println!("triangle_count: {}", triangle_count);
            println!("w: {:.}", triangle_width);
            println!("h: {:.}", triangle_height);
            frame_time_accumulator = 0f32;
        }

        // Multiply by 100 and clamp betwwn 0 and 255
        //let scaled_z = (z*50f32).min(255f32).max(0f32) as u32;

        // Switch to the depth buffer when A is pressed
        if (input_state.Gamepad.wButtons & XINPUT_GAMEPAD_A) != 0 {
            let int_depth_buffer: Vec<u32> = depth_buffer.iter().map(|f| {
                let channel = f.mul(50f32).min(255f32).max(0f32) as u32;
                channel << 16 | channel << 8 | channel
            }).collect();
            window
                .update_with_buffer(&int_depth_buffer, WIDTH as usize, HEIGHT as usize)
                .unwrap();
        } else {
            window
                .update_with_buffer(&buffer, WIDTH as usize, HEIGHT as usize)
                .unwrap();
        }

        if 8_000f32 < tx.abs() {
            t += tx / 1_000_000f32;
        }
        frame_number += 1;
    }
}

