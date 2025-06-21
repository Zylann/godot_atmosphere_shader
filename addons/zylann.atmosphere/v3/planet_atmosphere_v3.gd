@tool
class_name PlanetAtmosphereV3
extends Node3D


@export var compositor_effect: PlanetAtmosphereEffect

var _directional_light: DirectionalLight3D
#var _world_environment: WorldEnvironment


func _process(delta: float) -> void:
	if is_null_or_invalid(_directional_light):
		_directional_light = find_directional_light(get_viewport())

	if compositor_effect == null:
		#if is_null_or_invalid(_world_environment):
		# https://github.com/godotengine/godot-proposals/issues/9815
		var world_environment := find_world_environment(get_viewport())
		if world_environment != null:
			var compositor := world_environment.compositor
			if compositor == null:
				compositor = Compositor.new()
				world_environment.compositor = compositor
			var effects := compositor.compositor_effects
			for effect in effects:
				var comp_effect := effect as PlanetAtmosphereEffect
				if comp_effect != null:
					compositor_effect = comp_effect
					break
			if compositor_effect == null:
				compositor_effect = PlanetAtmosphereEffect.new()
				effects.append(compositor_effect)
				compositor.compositor_effects = effects
	
	if compositor_effect != null:
		if _directional_light != null:
			compositor_effect.sun_direction = -_directional_light.global_transform.basis.z
			compositor_effect.model_transform = global_transform


static func is_null_or_invalid(o) -> bool:
	return o == null or not is_instance_valid(o)


static func find_world_environment(parent: Node) -> WorldEnvironment:
	for i in parent.get_child_count():
		var node := parent.get_child(i)
		var wenv := node as WorldEnvironment
		if wenv != null:
			return wenv
		if node is Viewport:
			continue
		wenv = find_world_environment(node)
		if wenv != null:
			return wenv
	return null


static func find_directional_light(parent: Node) -> DirectionalLight3D:
	for i in parent.get_child_count():
		var node := parent.get_child(i)
		var light := node as DirectionalLight3D
		if light != null:
			return light
		if node is Viewport:
			continue
		light = find_directional_light(node)
		if light != null:
			return light
	return null
