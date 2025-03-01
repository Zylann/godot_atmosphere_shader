shader_type canvas_item;
render_mode blend_disabled;

uniform int u_frame_index;
// Interesting note: `filter_linear_mipmap` causes the temporal result to be as blurry as the
// viewport, even though we are sampling at subpixel offsets over time...
// This shader is for raymarching anyways, but it's important to keep in mind.
uniform samplerCube u_cubemap : filter_linear;

uniform vec2 u_viewport_size;
uniform mat4 u_inv_projection_matrix;
uniform mat4 u_inv_view_matrix;


void fragment() {
	ivec2 temporal_offset_px = ivec2(u_frame_index & 3, (u_frame_index / 4) & 3);
//	temporal_offset_px.x = 3 - temporal_offset_px.x;
//	temporal_offset_px.y = 3 - temporal_offset_px.y;
	vec2 screen_uv = 
		(SCREEN_UV * u_viewport_size + vec2(temporal_offset_px) * 0.25) / u_viewport_size;
	
	// Test
	vec3 ndc = vec3(screen_uv * 2.0 - 1.0, 1.0);
	vec4 view_coords = u_inv_projection_matrix * vec4(ndc, 1.0);
	vec3 forward = normalize(view_coords.xyz);
	vec3 forward_world = (u_inv_view_matrix * vec4(forward, 0.0)).xyz;
	vec3 output = texture(u_cubemap, forward_world).rrr;
	
	COLOR = vec4(output, 1.0);
//	ALBEDO = vec3(step(texture(u_cubemap, forward_world).r, 0.1), 0.0, 0.0);
	
	// TODO Render from screen_uv
	
	// R: transmittance
	// G: light
	// B: whatever
	// A: not used
}
