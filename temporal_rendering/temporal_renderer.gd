extends Node

#const TemporalBufferShader = preload("./temporal_buffer.gdshader")
#const TemporalRenderShader = preload("./temporal_render.gdshader")
const TemporalBufferShader = \
	preload("res://addons/zylann.atmosphere/shaders/cloud_temporal_buffer.gdshader")
const TemporalRenderShader = \
	preload("res://addons/zylann.atmosphere/shaders/cloud_temporal_render.gdshader")
const DebugDisplayResultShader = preload("./display_result.gdshader")

var _prev_view_proj_matrix := Projection()
var _prev_camera_transform := Transform3D()
var _prev_inv_proj_matrix := Projection()
var _prev_inv_view_matrix := Transform3D()
var _prev_planet_center_viewspace := Vector3()

class Buffer:
	var viewport : SubViewport
	var shader_material : ShaderMaterial

class Render:
	var viewport : SubViewport
	var shader_material : ShaderMaterial

const DEBUG_DISABLED = 0
const DEBUG_RENDER = 1
const DEBUG_BUFFER = 2

var _buffer0 := Buffer.new()
var _buffer1 := Buffer.new()
var _render := Render.new()
var _quad_mesh : QuadMesh
var _debug_texture_rect : TextureRect
var _debug_mode := DEBUG_DISABLED

func get_buffer_texture() -> Texture2D:
	return _buffer1.viewport.get_texture()


func _init():
	_quad_mesh = QuadMesh.new()
	_quad_mesh.size = Vector2(2, 2)
	_quad_mesh.flip_faces = true


func _ready():
	_setup_render(_render)
	
	_setup_buffer(_buffer0, 0)
	_setup_buffer(_buffer1, 1)


func _setup_buffer(b: Buffer, index: int):
	b.viewport = SubViewport.new()
	b.viewport.size = get_viewport().size
	b.viewport.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
	b.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	b.viewport.transparent_bg = false
	b.viewport.disable_3d = true
	b.viewport.name = str("BufferViewport", index)
	add_child(b.viewport)
	
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.0, 0.0, 0.0)
	
	var world_env := WorldEnvironment.new()
	world_env.environment = env
	b.viewport.add_child(world_env)
	
	b.shader_material = ShaderMaterial.new()
	b.shader_material.set_shader_parameter(&"u_render_texture", _render.viewport.get_texture())
	b.shader_material.shader = TemporalBufferShader
	
	var ci := ColorRect.new()
	ci.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ci.material = b.shader_material
	
	b.viewport.add_child(ci)


func _setup_render(r: Render):
	r.viewport = SubViewport.new()
	r.viewport.size = get_viewport().size / 4
	r.viewport.render_target_clear_mode = SubViewport.CLEAR_MODE_NEVER
	r.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	r.viewport.transparent_bg = false
	r.viewport.disable_3d = true
	r.viewport.name = "RenderViewport"
	add_child(r.viewport)
	
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.0, 0.0, 0.0)
	
	var world_env := WorldEnvironment.new()
	world_env.environment = env
	r.viewport.add_child(world_env)
	
	r.shader_material = ShaderMaterial.new()
	r.shader_material.shader = TemporalRenderShader
	
	var ci := ColorRect.new()
	ci.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ci.material = r.shader_material
	
	r.viewport.add_child(ci)


func _update_atmosphere_params():
	var atmosphere : Node3D = get_parent()
	var shader : Shader = atmosphere.custom_shader

	var params := shader.get_shader_uniform_list()
	for param in params:
		var param_name : String = param.name
		var value = atmosphere.get_shader_parameter(param_name)
		#_buffer0.shader_material.set_shader_parameter(param_name, value)
		_buffer1.shader_material.set_shader_parameter(param_name, value)
		_render.shader_material.set_shader_parameter(param_name, value)
	
	var camera := get_viewport().get_camera_3d()
	var world_to_view := camera.global_transform.inverse()
	var planet_center_world := atmosphere.global_position
	var planet_center_view := world_to_view * planet_center_world
	var sun : Node3D = atmosphere.get_node(atmosphere.sun_path)
	var sun_dir_world := sun.global_transform.basis.z
	var sun_dir_view = world_to_view.basis * sun_dir_world
	_render.shader_material.set_shader_parameter(&"u_planet_center_viewspace", planet_center_view)
	_render.shader_material.set_shader_parameter(&"u_sun_direction_viewspace", sun_dir_view)

	_buffer1.shader_material.set_shader_parameter(&"u_planet_center_viewspace", planet_center_view)
	_buffer1.shader_material.set_shader_parameter(
		&"u_prev_frame_inv_projection_matrix", _prev_inv_proj_matrix)
	_buffer1.shader_material.set_shader_parameter(
		&"u_prev_frame_inv_view_matrix", _prev_inv_view_matrix)
	_buffer1.shader_material.set_shader_parameter(
		&"u_prev_frame_planet_center_viewspace", _prev_planet_center_viewspace)
	
	atmosphere.set_shader_parameter(&"u_cloud_temporal_buffer", get_buffer_texture())

	_prev_planet_center_viewspace = planet_center_view


func _unhandled_input(event: InputEvent):
	if event is InputEventKey:
		if event.pressed:
			match event.keycode:
				KEY_KP_0:
					_set_debug_mode(DEBUG_DISABLED)
				KEY_KP_1:
					_set_debug_mode(DEBUG_RENDER)
				KEY_KP_2:
					_set_debug_mode(DEBUG_BUFFER)


func _set_debug_mode(mode: int):
	_debug_mode = mode
	print("Setting debug mode to ", mode)
	
	if mode == DEBUG_DISABLED:
		if _debug_texture_rect != null:
			_debug_texture_rect.hide()
			_debug_texture_rect.texture = null
		return
	
	if _debug_texture_rect == null:
		_debug_texture_rect = TextureRect.new()
		var sm := ShaderMaterial.new()
		sm.shader = DebugDisplayResultShader
		_debug_texture_rect.material = sm
		_debug_texture_rect.flip_v = true
		add_child(_debug_texture_rect)

	_debug_texture_rect.show()


func _process(delta):
	_update_atmosphere_params()
	
	var camera := get_viewport().get_camera_3d()
	
	var view_matrix := camera.global_transform.inverse()
	
	var viewport : Viewport = get_viewport()
	# Need to use buffer aspect because it must be multiple of downscale size
	var aspect := Vector2(_buffer0.viewport.size).aspect()
	# TODO Use `Camera3D.get_projection_matrix()` in 4.2
	var proj_matrix := Projection.create_perspective(camera.fov, aspect, camera.near, camera.far)
	var view_proj_matrix := proj_matrix * Projection(view_matrix)
	
	var proj_matrix_inverse := proj_matrix.inverse()
	var view_matrix_inverse := view_matrix.inverse()
	
	var frame_index := Engine.get_frames_drawn()
	
#	_buffer1.camera.global_transform = _camera.global_transform
	_buffer1.shader_material.set_shader_parameter(
		&"u_prev_frame_view_projection_matrix", _prev_view_proj_matrix)
	_buffer1.shader_material.set_shader_parameter(&"u_frame_index", frame_index)
	# WHY IS SCREEN TEXTURE NEVER WORKING THE WAY I WANT
	_buffer1.shader_material.set_shader_parameter(
		&"u_prev_screen_texture", _buffer0.viewport.get_texture())
	_buffer1.shader_material.set_shader_parameter(&"u_inv_projection_matrix", proj_matrix_inverse)
	_buffer1.shader_material.set_shader_parameter(&"u_inv_view_matrix", view_matrix_inverse)
	_buffer1.shader_material.set_shader_parameter(
		&"u_viewport_size", Vector2(_buffer1.viewport.size))
	
	_buffer0.viewport.render_target_update_mode = SubViewport.UPDATE_DISABLED
	_buffer1.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	
	if _debug_texture_rect != null:
		if _debug_mode == DEBUG_BUFFER:
			_debug_texture_rect.texture = _buffer0.viewport.get_texture()
		else:
			_debug_texture_rect.texture = _render.viewport.get_texture()
	
	# No idea if this will render before or after we sample it...
	# In fact I'm sure it's already causing problems!!!
#	_render.camera.global_transform = _camera.global_transform
	_render.shader_material.set_shader_parameter(&"u_frame_index", frame_index)
	_render.shader_material.set_shader_parameter(&"u_inv_projection_matrix", proj_matrix_inverse)
	_render.shader_material.set_shader_parameter(&"u_inv_view_matrix", view_matrix_inverse)
	_render.shader_material.set_shader_parameter(&"u_viewport_size", Vector2(_render.viewport.size))
	
	# Swap buffers
	var temp := _buffer0
	_buffer0 = _buffer1
	_buffer1 = temp
	
#	_buffer0.viewport.get_texture().get_image() \
#		.save_png(str("capture/buffer_", str(frame_index).pad_zeros(4), ".png"))
#	_render.viewport.get_texture().get_image() \
#		.save_png(str("capture/render_", str(frame_index).pad_zeros(4), ".png"))
#	get_viewport().get_texture().get_image() \
#		.save_png(str("capture/final_", str(frame_index).pad_zeros(4), ".png"))
	
	_prev_view_proj_matrix = view_proj_matrix
	_prev_camera_transform = camera.global_transform
	_prev_inv_proj_matrix = proj_matrix.inverse()
	_prev_inv_view_matrix = view_matrix.inverse()
	
	var win_size = get_viewport().size
	if _ack_size != win_size:
		_ack_size = win_size
		_resize_targets(win_size)


# Doesn't work??
#func _notification(what):
#	if what == NOTIFICATION_WM_SIZE_CHANGED:
#		var size = get_viewport().size
#		print("Resize ", size)
#		_render.viewport.size = size / 4
#		_buffer0.viewport.size = size
#		_buffer1.viewport.size = size


var _ack_size := Vector2i()
func _resize_targets(size: Vector2i):
#	size /= 2
	# Keep target sizes multiple of downscale factor to keep pixels aligned.
	size = (size / 4) * 4
	print("Resize ", size)
	_render.viewport.size = size / 4
	_buffer0.viewport.size = size
	_buffer1.viewport.size = size
	# TODO For some reason changing viewports size creates huge visual glitches...
#	_buffer0.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
#	_buffer1.viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS


