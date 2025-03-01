shader_type canvas_item;
render_mode blend_disabled;

// TODO We can raise this eventually
#define CLOUDS_MAX_RAYMARCH_STEPS 128
//#define CLOUDS_USE_DETAIL
//#define CLOUDS_RAYMARCHED_LIGHTING
#include "include/cloud_funcs.gdshaderinc"

uniform int u_frame_index;
// Interesting note: `filter_linear_mipmap` causes the temporal result to be as blurry as the
// viewport, even though we are sampling at subpixel offsets over time...
// This shader is for raymarching anyways, but it's important to keep in mind.
//uniform samplerCube u_cubemap : filter_linear;

// TODO Need access to the main depth texture
//uniform sampler2D u_main_depth_texture;
uniform vec3 u_planet_center_viewspace;
uniform float u_sphere_depth_factor;
uniform vec3 u_sun_direction_viewspace;
uniform sampler2D u_blue_noise_texture : filter_nearest, repeat_enable;

uniform mat4 u_inv_projection_matrix;
uniform mat4 u_inv_view_matrix;
// TODO Viewport size not available in canvas shaders???
uniform vec2 u_viewport_size;


void fragment() {
	mat4 inv_projection_matrix = u_inv_projection_matrix;
	mat4 inv_view_matrix = u_inv_view_matrix;
	vec2 viewport_size = u_viewport_size;

	ivec2 temporal_offset_px = ivec2(u_frame_index & 3, (u_frame_index / 4) & 3);
//	temporal_offset_px.x = 3 - temporal_offset_px.x;
//	temporal_offset_px.y = 3 - temporal_offset_px.y;
	vec2 screen_uv = (SCREEN_UV * viewport_size + vec2(temporal_offset_px) * 0.25) / viewport_size;

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

	float atmosphere_radius = u_planet_radius + u_atmosphere_height;
	vec2 rs_atmo = ray_sphere(u_planet_center_viewspace, atmosphere_radius, ray_origin, ray_dir);

	if (rs_atmo.x != rs_atmo.y) {
		vec2 rs_ground =
			ray_sphere(u_planet_center_viewspace, u_planet_radius, ray_origin, ray_dir);
		float gd = 10000000.0;
		if (rs_ground.x != rs_ground.y) {
			gd = rs_ground.x;
		}
		linear_depth = mix(linear_depth, gd, u_sphere_depth_factor);

		vec3 sun_dir = u_sun_direction_viewspace;

		// TODO Might need to adjust this for quarter-res rendering
		// Blue noise doesn't have low-frequency patterns, it looks less "noisy"
		// http://momentsingraphics.de/BlueNoise.html
		vec2 jitter_uv = (viewport_size * 4.0) * screen_uv;// + vec2(TIME*1000.0, 0);
		float jitter = texelFetch(u_blue_noise_texture, ivec2(jitter_uv) & ivec2(0xff), 0).r;
//		jitter = 0.0;

		vec3 cloud = render_clouds_data(u_planet_center_viewspace, ray_origin,
			ray_dir, linear_depth, inv_view_matrix, sun_dir, jitter);

		COLOR = vec4(
			// Hack to get more resolution in dark areas and reduce banding.
			// It gets undone when read.
			sqrt(cloud.r), // Color
			cloud.g, // Alpha
			cloud.b, // Depth within the cloud layer, relative to first ray hit
			0.0);

	} else {
		COLOR = vec4(0.0);
	}
}
