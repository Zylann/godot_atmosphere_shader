shader_type canvas_item;
render_mode blend_disabled;

// Based on:
// https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn

// proj * view
// Where:
// proj = Projection.create_perspective(_camera.fov, aspect, _camera.near, _camera.far)
// view = _camera.global_transform.inverse()
uniform mat4 u_prev_frame_view_projection_matrix;

uniform int u_frame_index;
// TODO hint_screen_texture didn't work?? Had to use an additional viewport and alternate...
uniform sampler2D u_prev_screen_texture : repeat_disable, source_color;// : hint_screen_texture;
// TODO Sample at pixel center so we can use filter_linear
uniform sampler2D u_render_texture : source_color, filter_nearest;

uniform mat4 u_inv_view_matrix;
uniform mat4 u_inv_projection_matrix;
uniform vec2 u_viewport_size;

vec2 reproject_screen_position(vec2 screen_uv, mat4 proj_to_world) {
	// Sourced from:
	// https://www.gamedev.net/forums/topic/698511-temporal-reprojection-on-volumetric-cloud-rendering/
	
	vec4 ray_clip = vec4(2.0 * screen_uv - vec2(1.0), 
		// Assuming clouds are on the far plane. For now we can't really decide of a depth...
		// maybe we could if we somehow rendered depth of the first sample with high opacity?
		1.0, 1.0);
	vec4 view_to_world_pos = proj_to_world * ray_clip;
	view_to_world_pos /= view_to_world_pos.w;
	vec4 prev_ray_clip = u_prev_frame_view_projection_matrix * view_to_world_pos;
	prev_ray_clip /= prev_ray_clip.w;
	vec2 prev_frame_screen_uv = prev_ray_clip.xy * 0.5 + vec2(0.5);
	return prev_frame_screen_uv;
}

void fragment() {
	vec2 prev_frame_screen_uv = reproject_screen_position(
		SCREEN_UV, u_inv_view_matrix * u_inv_projection_matrix);
	
	float in_prev_frame = float(
		prev_frame_screen_uv.x >= 0.0 && prev_frame_screen_uv.y >= 0.0 && 
		prev_frame_screen_uv.x <= 1.0 && prev_frame_screen_uv.y <= 1.0);
//	in_prev_frame = 1.0;
	
	vec3 lowres_cloud = texture(u_render_texture, SCREEN_UV).rgb;
	
	vec3 output = mix(lowres_cloud, texture(u_prev_screen_texture, 
		vec2(prev_frame_screen_uv.x, prev_frame_screen_uv.y)).rgb, in_prev_frame);
//	ALBEDO = texture(u_prev_screen_texture, SCREEN_UV).rgb;
//	ALBEDO = vec3(0.0);
	
	ivec2 pos_px = ivec2(SCREEN_UV * u_viewport_size);
	int pxi = (pos_px.x & 3) + (pos_px.y & 3) * 4;
	if (((pxi - u_frame_index) & 15) == 0) {
//		ALBEDO = vec3(0.0, 1.0, 0.0);
		output = lowres_cloud;
	}
	
	COLOR = vec4(output, 1.0);
}

