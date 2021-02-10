tool
extends Spatial

const MODE_NEAR = 0
const MODE_FAR = 1
const SWITCH_MARGIN_RATIO = 1.1

const AtmosphereShader = preload("./planet_atmosphere.shader")

export var planet_radius := 1.0 setget set_planet_radius
export var atmosphere_height := 0.1 setget set_atmosphere_height
export(NodePath) var sun_path : NodePath setget set_sun_path
export var day_color0 := Color(0.29, 0.39, 0.92) setget set_day_color0
export var day_color1 := Color(0.76, 0.90, 1.0) setget set_day_color1
export var night_color0 := Color(0.15, 0.10, 0.33) setget set_night_color0
export var night_color1 := Color(0.0, 0.0, 0.0) setget set_night_color1
export var density := 0.2 setget set_density

var _far_mesh : CubeMesh
var _near_mesh : QuadMesh
var _mode := MODE_FAR
var _mesh_instance : MeshInstance


func _init():
	var material = ShaderMaterial.new()
	material.shader = AtmosphereShader
	_mesh_instance = MeshInstance.new()
	_mesh_instance.material_override = material
	_mesh_instance.cast_shadow = false
	add_child(_mesh_instance)

	_near_mesh = QuadMesh.new()
	_near_mesh.size = Vector2(2.0, 2.0)
	
	#_far_mesh = _create_far_mesh()
	_far_mesh = CubeMesh.new()
	_far_mesh.size = Vector3(1.0, 1.0, 1.0)

	_mesh_instance.mesh = _far_mesh
	
	_update_cull_margin()


func _ready():
	var mat = _mesh_instance.material_override
	mat.set_shader_param("u_planet_radius", planet_radius)
	mat.set_shader_param("u_atmosphere_height", atmosphere_height)
	mat.set_shader_param("u_clip_mode", false)
	mat.set_shader_param("u_day_color", day_color0)
	mat.set_shader_param("u_day_color", day_color1)
	mat.set_shader_param("u_night_color", night_color0)
	mat.set_shader_param("u_night_color", night_color1)


func set_planet_radius(new_radius: float):
	if planet_radius == new_radius:
		return
	planet_radius = new_radius
	_mesh_instance.material_override.set_shader_param("u_planet_radius", planet_radius)
	_update_cull_margin()


func _update_cull_margin():
	_mesh_instance.extra_cull_margin = planet_radius + atmosphere_height


func set_atmosphere_height(new_height: float):
	if atmosphere_height == new_height:
		return
	atmosphere_height = new_height
	_mesh_instance.material_override.set_shader_param("u_atmosphere_height", atmosphere_height)
	_update_cull_margin()


func set_sun_path(new_sun_path: NodePath):
	sun_path = new_sun_path


func set_day_color0(new_day_color0: Color):
	if day_color0 == new_day_color0:
		return
	day_color0 = new_day_color0
	_mesh_instance.material_override.set_shader_param("u_day_color0", day_color0)


func set_day_color1(new_day_color1: Color):
	if day_color1 == new_day_color1:
		return
	day_color1 = new_day_color1
	_mesh_instance.material_override.set_shader_param("u_day_color1", day_color1)


func set_night_color0(new_night_color0: Color):
	if night_color0 == new_night_color0:
		return
	night_color0 = new_night_color0
	_mesh_instance.material_override.set_shader_param("u_night_color0", night_color0)


func set_night_color1(new_night_color1: Color):
	if night_color1 == new_night_color1:
		return
	night_color1 = new_night_color1
	_mesh_instance.material_override.set_shader_param("u_night_color1", night_color1)


func set_density(new_density: float):
	if new_density == density:
		return
	density = new_density
	_mesh_instance.material_override.set_shader_param("u_density", density)


func _set_mode(mode: int):
	if mode == _mode:
		return
	_mode = mode

	var mat = _mesh_instance.material_override

	if _mode == MODE_NEAR:
		if OS.is_stdout_verbose():
			print("Switching ", name, " to near mode")
		# If camera is close enough, switch shader to near clip mode
		# otherwise it will pass through the quad
		mat.set_shader_param("u_clip_mode", true)
		_mesh_instance.mesh = _near_mesh
		_mesh_instance.transform = Transform()
		# TODO Sometimes there is a short flicker, figure out why

	else:
		if OS.is_stdout_verbose():
			print("Switching ", name, " to far mode")
		mat.set_shader_param("u_clip_mode", false)
		_mesh_instance.mesh = _far_mesh


func _process(_delta):
	var cam_pos := Vector3()
	var cam_near := 0.1
	
	var cam = get_viewport().get_camera()

	if cam != null:
		cam_pos = cam.global_transform.origin
		cam_near = cam.near
		
	elif Engine.editor_hint:
		# Getting the camera in editor is freaking awkward so let's hardcode it...
		cam_pos = global_transform.origin \
			+ Vector3(10.0 * (planet_radius + atmosphere_height + cam_near), 0, 0)

	# 1.75 is an approximation of sqrt(3), because the far mesh is a cube and we have to take
	# the largest distance from the center into account
	var atmo_clip_distance : float = \
		1.75 * (planet_radius + atmosphere_height + cam_near) * SWITCH_MARGIN_RATIO
	
	# Detect when to switch modes.
	# we always switch modes while already being slightly away from the quad, to avoid flickering
	var d := global_transform.origin.distance_to(cam_pos)
	var is_near := d < atmo_clip_distance
	if is_near:
		_set_mode(MODE_NEAR)
	else:
		_set_mode(MODE_FAR)

	if _mode == MODE_FAR:
		_mesh_instance.scale = \
			Vector3(atmo_clip_distance, atmo_clip_distance, atmo_clip_distance)
	
	# Lazily avoiding the node referencing can of worms.
	# Not very efficient but I assume there won't be many atmospheres in the game.
	# In Godot 4 it could be replaced by caching the object ID in some way
	if has_node(sun_path):
		var sun = get_node(sun_path)
		if sun is Spatial:
			_mesh_instance.material_override.set_shader_param(
				"u_sun_position", sun.global_transform.origin)
