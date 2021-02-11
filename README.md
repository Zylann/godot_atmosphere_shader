Atmosphere shader for Godot Engine
====================================

This contains a simple atmosphere shader usable on planets. I developped this for a tech demo I'm working on, so I thought of making it an addon.

- Not realistic, but fast (no nested for loops)
- Gradients of two colors for each side of the planet
- Can be seen from inside like regular fog if you want to land on the planet
- Switches to a cube mesh when seen from far away so multiple atmospheres can be drawn at lower cost

I'm interested in having a realistic version of the shader but it's not a priority at the moment.


How to use
-----------

Copy the contents of `zylann.atmosphere` under the `res://addons` folder of your project. There is no need to activate a plugin.

Drag and drop `planet_atmosphere.tscn` as child of your planet node. In the inspector, give it the same radius, and choose a height. You can tweak colors by expanding the `shader_params` category. Depending on the size of your planet, you may also have to tune density since light will have to travel larger distances through it.
