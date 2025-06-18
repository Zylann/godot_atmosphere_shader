@tool
class_name PlanetAtmosphereEffect
extends CompositorEffect

# Refs:
# https://www.youtube.com/watch?v=hqhWR0CxZHA

# Pass 1: render clouds at quarter resolution
# Pass 2: upscale clouds
# Pass 3: blur clouds
# Pass 4: blend clouds

var _dirty : bool = true
var _mutex : Mutex

var _rd : RenderingDevice
var _shader_rid : RID
var _pipeline_rid : RID
var _linear_sampler : RID
var _nearest_sampler : RID
var _params_ubo : RID
var _cam_params_ubo : RID

const _shader_file = preload("./effect.glsl")
const _cloud_coverage_cubemap = preload("res://tests/cloud_coverage.png")
const _cloud_shape_texture = \
	preload("res://addons/zylann.atmosphere/demo/cloud_shape_texture3d.tres")
const _blue_noise_texture = preload("res://addons/zylann.atmosphere/blue_noise.png")


func _init() -> void:
	_mutex = Mutex.new()
	
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	_rd = RenderingServer.get_rendering_device()
	
	RenderingServer.call_on_render_thread(_init_render)


func _init_render() -> void:
	var ss = RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	ss.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_w = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	_nearest_sampler = _rd.sampler_create(ss)
	
	ss = RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss.repeat_w = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	_linear_sampler = _rd.sampler_create(ss)
	
	var params_f32 := _make_params_f32()
	_params_ubo = _rd.uniform_buffer_create(params_f32.size() * 4, params_f32.to_byte_array())
	
	var cam_params_f32 := _make_camera_params_f32(null)
	_cam_params_ubo = _rd.uniform_buffer_create(
		cam_params_f32.size() * 4, 
		cam_params_f32.to_byte_array()
	)


func _clear_render() -> void:
	if _shader_rid.is_valid():
		_rd.free_rid(_shader_rid)
		_shader_rid = RID()

	if _nearest_sampler.is_valid():
		_rd.free_rid(_nearest_sampler)
		_nearest_sampler = RID()
	
	if _linear_sampler.is_valid():
		_rd.free_rid(_linear_sampler)
		_linear_sampler = RID()
	
	if _params_ubo.is_valid():
		_rd.free_rid(_params_ubo)
		_params_ubo = RID()
	
	if _cam_params_ubo.is_valid():
		_rd.free_rid(_cam_params_ubo)
		_cam_params_ubo = RID()


func _notification(what: int) -> void:
	match what:
		NOTIFICATION_PREDELETE:
			# TODO Can't call our own methods on cleanup... so we have code repetition...
			# https://github.com/godotengine/godot-proposals/issues/11774
			#_clear_render()
			
			_rd.free_rid(_shader_rid)
			_rd.free_rid(_nearest_sampler)
			_rd.free_rid(_linear_sampler)
			_rd.free_rid(_params_ubo)
			_rd.free_rid(_cam_params_ubo)


func _update_shader() -> bool:
	if not _dirty:
		return true
	_dirty = false
	
	# Does any of this actually need mutex lock?
	
	var shader_spirv := _shader_file.get_spirv()
	
	if shader_spirv.compile_error_compute != "":
		return false
	
	_shader_rid = _rd.shader_create_from_spirv(shader_spirv)
	if not _shader_rid.is_valid():
		return false
	
	_pipeline_rid = _rd.compute_pipeline_create(_shader_rid)
	return _pipeline_rid.is_valid()


func _render_callback(p_effect_callback_type: int, p_render_data: RenderData) -> void:
	if _rd == null:
		return
	
	# Looks redundant, but users might be able to change that in the UI
	if p_effect_callback_type != EFFECT_CALLBACK_TYPE_POST_TRANSPARENT:
		return

	if not _update_shader():
		return
	
	# Get our render scene buffers object, this gives us access to our render buffers.
	# Note that implementation differs per renderer hence the need for the cast.
	var render_scene_buffers : RenderSceneBuffersRD = p_render_data.get_render_scene_buffers()
	if render_scene_buffers == null:
		return

	# Get our render size, this is the 3D render resolution!
	var size := render_scene_buffers.get_internal_size()
	if size.x == 0 and size.y == 0:
		return
	
	# We can use a compute shader here.
	var x_groups := ceildiv(size.x, 8)
	var y_groups := ceildiv(size.y, 8)
	var z_groups := 1

	var time_seconds := Time.get_ticks_msec() / 1000.0

	var scene_data := p_render_data.get_render_scene_data()
	var camera_transform := scene_data.get_cam_transform()
	var world_to_view := camera_transform.inverse()
	var planet_center_world := Vector3()
	var planet_center_viewspace := world_to_view * planet_center_world
	var sphere_depth_factor := 0.0
	var sun_center_world := Vector3(0.0, 1000.0, 0.0)
	var sun_center_viewspace := world_to_view * sun_center_world
	
	# Faster than UBO but typically limited in size (128 bytes minimum).
	# Also needs to be aligned to 16 bytes
	var push_constant: PackedFloat32Array = PackedFloat32Array()

	push_constant.push_back(size.x)
	push_constant.push_back(size.y)
	push_constant.push_back(time_seconds)
	push_constant.push_back(0.0)

	push_constant.push_back(planet_center_viewspace.x)
	push_constant.push_back(planet_center_viewspace.y)
	push_constant.push_back(planet_center_viewspace.z)
	push_constant.push_back(sphere_depth_factor)

	push_constant.push_back(sun_center_viewspace.x)
	push_constant.push_back(sun_center_viewspace.y)
	push_constant.push_back(sun_center_viewspace.z)
	push_constant.push_back(0.0)
	
	var cam_params_f32 := _make_camera_params_f32(scene_data)
	var cam_params_bytes := cam_params_f32.to_byte_array()
	_rd.buffer_update(_cam_params_ubo, 0, cam_params_bytes.size(), cam_params_bytes)
	
	# # Loop through views just in case we're doing stereo rendering.
	var view_count := render_scene_buffers.get_view_count()
	for view_index in view_count:
		var color_image := render_scene_buffers.get_color_layer(view_index)
		var color_image_uniform := RDUniform.new()
		color_image_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
		color_image_uniform.binding = 0
		color_image_uniform.add_id(color_image)
		
		var depth_image := render_scene_buffers.get_depth_layer(view_index)
		var depth_texture_uniform := RDUniform.new()
		depth_texture_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		depth_texture_uniform.binding = 1
		depth_texture_uniform.add_id(_nearest_sampler)
		depth_texture_uniform.add_id(depth_image)
		
		var cloud_coverage_cubemap_rd := \
			RenderingServer.texture_get_rd_texture(_cloud_coverage_cubemap.get_rid())
		var cloud_coverage_cubemap_uniform := RDUniform.new()
		cloud_coverage_cubemap_uniform.uniform_type = \
			RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		cloud_coverage_cubemap_uniform.binding = 2
		cloud_coverage_cubemap_uniform.add_id(_linear_sampler)
		cloud_coverage_cubemap_uniform.add_id(cloud_coverage_cubemap_rd)
		
		var cloud_shape_texture_rd := \
			RenderingServer.texture_get_rd_texture(_cloud_shape_texture.get_rid())
		var cloud_shape_texture_uniform := RDUniform.new()
		cloud_shape_texture_uniform.uniform_type = \
			RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		cloud_shape_texture_uniform.binding = 3
		cloud_shape_texture_uniform.add_id(_linear_sampler)
		cloud_shape_texture_uniform.add_id(cloud_shape_texture_rd)
		
		var blue_noise_texture_rd := \
			RenderingServer.texture_get_rd_texture(_blue_noise_texture.get_rid())
		var blue_noise_texture_uniform := RDUniform.new()
		blue_noise_texture_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		blue_noise_texture_uniform.binding = 4
		blue_noise_texture_uniform.add_id(_linear_sampler)
		blue_noise_texture_uniform.add_id(blue_noise_texture_rd)
		
		var params_uniform := RDUniform.new()
		params_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
		params_uniform.binding = 5
		params_uniform.add_id(_params_ubo)

		var cam_params_uniform := RDUniform.new()
		cam_params_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
		cam_params_uniform.binding = 6
		cam_params_uniform.add_id(_cam_params_ubo)
		
		var uniform_set_items := [
			color_image_uniform,
			depth_texture_uniform,
			cloud_coverage_cubemap_uniform,
			cloud_shape_texture_uniform,
			blue_noise_texture_uniform,
			params_uniform,
			cam_params_uniform
		]
		
		var uniform_set := UniformSetCacheRD.get_cache(_shader_rid, 0, uniform_set_items)
		
		var compute_list := _rd.compute_list_begin()
		_rd.compute_list_bind_compute_pipeline(compute_list, _pipeline_rid)
		_rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
		var push_constant_bytes = push_constant.to_byte_array()
		_rd.compute_list_set_push_constant(
			compute_list, 
			push_constant_bytes, 
			push_constant_bytes.size()
		)
		_rd.compute_list_dispatch(compute_list, x_groups, y_groups, z_groups)
		_rd.compute_list_end()


func _make_params_f32() -> PackedFloat32Array:
	var world_to_model := Transform3D()
	var planet_radius := 100.0
	var atmosphere_height := 20.0
	var cloud_density_scale := 50.0
	var cloud_bottom := 0.2
	var cloud_top := 0.5
	var cloud_blend := 0.5
	var cloud_shape_invert := 0.0
	var cloud_coverage_bias := 0.0
	var cloud_coverage_rotation_x := Vector2(1.0, 0.0)
	var cloud_coverage_rotation_y := Vector2(0.0, 1.0)
	var cloud_shape_factor := 0.5
	var cloud_shape_scale := 0.1
	
	var params_f32 := PackedFloat32Array()
	_encode_transform_to_mat4(params_f32, world_to_model)

	params_f32.append(cloud_coverage_rotation_x.x)
	params_f32.append(cloud_coverage_rotation_x.y)
	params_f32.append(cloud_coverage_rotation_y.x)
	params_f32.append(cloud_coverage_rotation_y.y)

	params_f32.append(planet_radius)
	params_f32.append(atmosphere_height)
	params_f32.append(cloud_density_scale)
	params_f32.append(cloud_bottom)
	params_f32.append(cloud_top)
	params_f32.append(cloud_blend)
	params_f32.append(cloud_coverage_bias)
	params_f32.append(cloud_shape_invert)
	params_f32.append(cloud_shape_factor)
	params_f32.append(cloud_shape_scale)
	
	params_f32.append(0.0)
	params_f32.append(0.0)
	
	return params_f32


func _make_camera_params_f32(sd: RenderSceneData) -> PackedFloat32Array:
	var projection := sd.get_cam_projection() if sd != null else Projection()
	var inv_view := sd.get_cam_transform() if sd != null else Transform3D()
	var inv_projection := projection.inverse()
	
	var data_f32 := PackedFloat32Array()
	
	_encode_transform_to_mat4(data_f32, inv_view)
	_encode_projection_to_mat4(data_f32, inv_projection)
	
	return data_f32


static func _encode_transform_to_mat4(dst: PackedFloat32Array, t: Transform3D) -> void:
	var b := t.basis
	var o := t.origin

	dst.append(b.x.x)
	dst.append(b.x.y)
	dst.append(b.x.z)
	dst.append(0.0)
	
	dst.append(b.y.x)
	dst.append(b.y.y)
	dst.append(b.y.z)
	dst.append(0.0)

	dst.append(b.z.x)
	dst.append(b.z.y)
	dst.append(b.z.z)
	dst.append(0.0)

	dst.append(o.x)
	dst.append(o.y)
	dst.append(o.z)
	dst.append(1.0)


static func _encode_projection_to_mat4(dst: PackedFloat32Array, p: Projection) -> void:
	dst.append(p.x.x)
	dst.append(p.x.y)
	dst.append(p.x.z)
	dst.append(p.x.w)

	dst.append(p.y.x)
	dst.append(p.y.y)
	dst.append(p.y.z)
	dst.append(p.y.w)

	dst.append(p.z.x)
	dst.append(p.z.y)
	dst.append(p.z.z)
	dst.append(p.z.w)

	dst.append(p.w.x)
	dst.append(p.w.y)
	dst.append(p.w.z)
	dst.append(p.w.w)


static func ceildiv(x: int, d: int) -> int:
	assert(d > 0)
	if x > 0:
		return (x + d - 1) / d
	else:
		return x / d
