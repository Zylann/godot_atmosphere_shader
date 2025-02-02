Changelog
============

This is a high-level changelog for each released versions of the plugin.
For a more detailed list of past and incoming changes, see the commit history.

0.5 (dev)
------

- Enabled reverse-Z handling by default, so Godot 4.3 or later is preferred onwards.
- `NoiseCubemap` data is no longer saved to the resource file. This was unnecessary because the texture is procedural.


0.4
------

- Added `force_fullscreen` option to allow previewing the inside of the atmosphere in the editor
- Added atmosphere ambient color to v2 atmosphere so nights are no longer pitch black
- Added `*_shader_parameter()` methods, deprecated `*_shader_param` methods
- Slightly improved clouds alpha blending
- Decoupled alpha from color in v2 atmosphere so it no longer stops rendering in the dark side of planets
- Fixed properties list not updating when setting a different shader


0.3
----

- Added more realistic atmosphere model based on Sebastian Lague's Coding Adventure
- Added automatic optical depth baking using a viewport (Vulkan renderer not needed)
- Added raymarched and animated clouds (current version is imperfect due to tradeoffs)
- Added NoiseCubemap resource to generate procedural cloud coverage textures
- Added small demo


0.2
----

This version requires Godot 4.0 or later.

- Added option to blend ground depth with a sphere, to hide precision lost at high distances


0.1 (godot3 branch)
---------------------

- Initial release in Godot 3.2
