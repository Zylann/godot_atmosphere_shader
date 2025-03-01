shader_type canvas_item;
render_mode blend_disabled;

#include "include/util.gdshaderinc"

// Based on:
// https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn

// proj * view
// Where:
// proj = Projection.create_perspective(_camera.fov, aspect, _camera.near, _camera.far)
// view = _camera.global_transform.inverse()
uniform mat4 u_prev_frame_view_projection_matrix;
uniform mat4 u_prev_frame_inv_view_matrix;
uniform mat4 u_prev_frame_inv_projection_matrix;
uniform vec3 u_prev_frame_planet_center_viewspace;

uniform int u_frame_index;
// TODO hint_screen_texture didn't work?? Had to use an additional viewport and alternate...
uniform sampler2D u_prev_screen_texture : repeat_disable;// : hint_screen_texture;
// TODO How can I use filter_linear?
// It could help providing smooth estimation when needed, but even when sampling at the middle of
// pixels, it didn't work well...
uniform sampler2D u_render_texture : repeat_disable, filter_nearest;

uniform float u_planet_radius = 1.0;
uniform float u_atmosphere_height = 0.1;
uniform vec3 u_planet_center_viewspace;
uniform float u_sphere_depth_factor = 0.0;
uniform float u_cloud_bottom = 0.3;
uniform float u_cloud_top = 0.6;

uniform mat4 u_inv_view_matrix;
uniform mat4 u_inv_projection_matrix;
// TODO Viewport size not available in canvas shaders???
uniform vec2 u_viewport_size;

vec2 reproject_screen_position(vec2 screen_uv, mat4 proj_to_world, float nonlinear_depth) {
	// Sourced from:
	// https://www.gamedev.net/forums/topic/698511-temporal-reprojection-on-volumetric-cloud-rendering/

	// Note, nonlinear_depth would be 1.0 if we assume clouds are on the far plane.

	vec4 ray_clip = vec4(2.0 * screen_uv - vec2(1.0), nonlinear_depth, 1.0);
	vec4 view_to_world_pos = proj_to_world * ray_clip;
	view_to_world_pos /= view_to_world_pos.w;

	vec4 prev_ray_clip = u_prev_frame_view_projection_matrix * view_to_world_pos;
	prev_ray_clip /= prev_ray_clip.w;
	vec2 prev_frame_screen_uv = prev_ray_clip.xy * 0.5 + vec2(0.5);
	return prev_frame_screen_uv;
}

vec2 get_ray(
	vec2 screen_uv,
	mat4 inv_projection_matrix,
	mat4 inv_view_matrix,
	vec3 planet_center_viewspace,
	out vec3 out_ray_dir
	// Note, we could use `out` vars, but make sure they are always assigned, as it seems otherwise
	// we get undefined behaviors that don't even cause shader compiling errors or warnings...
) {
	// TODO Need access to depth
	float nonlinear_depth = 1.0;//texture(depth_texture, screen_uv).x;
	// Note, we assume Vulkan here, which has NDC in 0..1 (OpenGL is -1..1)
	vec3 ndc = vec3(screen_uv * 2.0 - 1.0, nonlinear_depth);
	vec4 view_coords = inv_projection_matrix * vec4(ndc, 1.0);
	//view_coords.xyz /= view_coords.w;
	//float linear_depth = -view_coords.z; // Not what I want because it changes when looking around
	vec4 world_coords = inv_view_matrix * view_coords;
	vec3 pos_world = world_coords.xyz / world_coords.w;
	vec3 cam_pos_world = (inv_view_matrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	// I wonder if there is a faster way to get to that distance...
	float linear_depth = distance(cam_pos_world, pos_world);

	// We'll evaluate the atmosphere in view space
	vec3 ray_origin = vec3(0.0, 0.0, 0.0);
	vec3 ray_dir = normalize(view_coords.xyz - ray_origin);
	out_ray_dir = ray_dir;

	float atmosphere_radius = u_planet_radius + u_atmosphere_height;
	vec2 rs_atmo = ray_sphere(planet_center_viewspace, atmosphere_radius, ray_origin, ray_dir);

	// TODO if we run this shader in a double-clip scenario,
	// we have to account for the near and far clips properly, so they can be composed seamlessly

	if (rs_atmo.x != rs_atmo.y) {
		float t_begin = max(rs_atmo.x, 0.0);
		float t_end = max(rs_atmo.y, 0.0);

		vec2 rs_ground = ray_sphere(planet_center_viewspace, u_planet_radius, ray_origin, ray_dir);
		float gd = 10000000.0;
		if (rs_ground.x != rs_ground.y) {
			gd = rs_ground.x;
		}
		linear_depth = mix(linear_depth, gd, u_sphere_depth_factor);

		t_end = min(t_end, linear_depth);

		float clouds_bottom = u_planet_radius + u_cloud_bottom * u_atmosphere_height;
		float clouds_top = u_planet_radius + u_cloud_top * u_atmosphere_height;

		vec2 rs_clouds_top = ray_sphere(planet_center_viewspace, clouds_top, ray_origin, ray_dir);

		if (rs_clouds_top.x != rs_clouds_top.y) {
			vec2 rs_clouds_bottom =
				ray_sphere(planet_center_viewspace, clouds_bottom, ray_origin, ray_dir);

			vec2 cloud_rs = rs_clouds_top;
			cloud_rs.x = max(cloud_rs.x, 0.0);
			cloud_rs.y = min(cloud_rs.y, linear_depth);

			if (cloud_rs.x < linear_depth
				// Don't compute clouds when opaque stuff occludes them,
				// when under the clouds layer.
				// This saves 0.5ms in ground views on a 1060
				&& (linear_depth > rs_clouds_bottom.y || rs_clouds_bottom.x > 0.0)
			) {
				return cloud_rs;
			}
		}
	}
	return vec2(0.0);
}

//vec4 texture_linear(sampler2D tex, vec2 uv) {
//	vec2 ts = vec2(textureSize(tex, 0));
//	vec2 ps = 1.0 / ts;
//	vec4 c00 = texture(tex, vec2(uv.x, uv.y));
//	vec4 c10 = texture(tex, vec2(uv.x + ps.x, uv.y));
//	vec4 c01 = texture(tex, vec2(uv.x, uv.y + ps.y));
//	vec4 c11 = texture(tex, vec2(uv.x + ps.x, uv.y + ps.y));
//	vec2 f = uv * ts - floor(uv * ts);
//	vec4 c = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
//	return c;
//}

vec2 get_prev_frame_uv_v1(vec2 screen_uv) {
	vec3 lowres_cloud = texture(u_render_texture, screen_uv).rgb;

	vec3 ray_dir_viewspace;
	vec2 rs = get_ray(screen_uv, u_inv_projection_matrix, u_inv_view_matrix,
		u_planet_center_viewspace, ray_dir_viewspace);
	float t_begin = rs.x;
	float t_end = rs.y;

	float cloud_depth = mix(t_begin, t_end, lowres_cloud.b);

	// TODO This is very crude, there should be a smarter way to do this?
	vec4 tmp = inverse(u_inv_projection_matrix) * vec4(cloud_depth * ray_dir_viewspace, 1.0);
	float nonlinear_cloud_depth = tmp.z / tmp.w;
	//nonlinear_cloud_depth = 1.0;

	vec2 prev_frame_screen_uv = reproject_screen_position(
		screen_uv, u_inv_view_matrix * u_inv_projection_matrix, nonlinear_cloud_depth);

	return prev_frame_screen_uv;
}

vec2 get_prev_frame_uv_v2(vec2 screen_uv) {
	vec3 lowres_cloud = texture(u_render_texture, screen_uv).rgb;

//	vec2 prev_frame_screen_far_uv = reproject_screen_position(
//		screen_uv, u_inv_view_matrix * u_inv_projection_matrix, 1.0);
	vec2 prev_frame_screen_far_uv = screen_uv;

	float prev_depth_param = texture(u_prev_screen_texture, prev_frame_screen_far_uv).b;
//	vec3 lowres_cloud_f = texture_linear(u_render_texture, screen_uv).rgb;

	vec3 ray_dir_viewspace;
	vec2 rs = get_ray(
		prev_frame_screen_far_uv,
		u_prev_frame_inv_projection_matrix,
		u_prev_frame_inv_view_matrix,
		u_prev_frame_planet_center_viewspace,
		ray_dir_viewspace);
	float t_begin = rs.x;
	float t_end = rs.y;

	float cloud_depth = mix(t_begin, t_end, prev_depth_param);

//	ray_dir_viewspace = vec3(0.0, 0.0, -1.0);
	// TODO This is very crude, there should be a smarter way to do this?
	vec4 tmp = inverse(u_inv_projection_matrix) * vec4(cloud_depth * ray_dir_viewspace, 1.0);
	float nonlinear_cloud_depth = tmp.z / tmp.w;
	//nonlinear_cloud_depth = 1.0;

	vec2 prev_frame_screen_uv = reproject_screen_position(
		screen_uv, u_inv_view_matrix * u_inv_projection_matrix, nonlinear_cloud_depth);

	return prev_frame_screen_uv;
}

void fragment() {
	vec3 lowres_cloud = texture(u_render_texture, SCREEN_UV).rgb;

	vec2 prev_frame_screen_uv = get_prev_frame_uv_v1(SCREEN_UV);

	float in_prev_frame = float(
		prev_frame_screen_uv.x >= 0.0 && prev_frame_screen_uv.y >= 0.0 &&
		prev_frame_screen_uv.x <= 1.0 && prev_frame_screen_uv.y <= 1.0);
//	in_prev_frame = 0.0;

//	float b1 = texture(u_prev_screen_texture, SCREEN_UV).b;
//	if((b1 == 0.0 || b1 == 1.0) != (lowres_cloud.b == 0.0 || lowres_cloud.b == 1.0)) {
//		in_prev_frame = 0.0;
//	}

	vec3 cloud = mix(lowres_cloud, texture(u_prev_screen_texture, prev_frame_screen_uv).rgb,
		in_prev_frame);

	ivec2 pos_px = ivec2(SCREEN_UV * u_viewport_size);
	int pxi = (pos_px.x & 3) + (pos_px.y & 3) * 4;
	if (((pxi - u_frame_index) & 15) == 0) {
		cloud = lowres_cloud;
	}

	COLOR = vec4(cloud.r, cloud.g, cloud.b, 1.0);
//	COLOR = vec4(cloud.r*0.0, cloud.g*0.0, fract(cloud_depth*0.1), 1.0);
//	if (t_begin == t_end) {
//	    COLOR = vec4(0.5, 0.0, 0.0, 1.0);
//	}
}
