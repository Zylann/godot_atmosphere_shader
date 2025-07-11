Atmosphere shader for Godot Engine 4.3+
=======================================

This contains a simple atmosphere shader usable on planets. I developped this for a tech demo I'm working on, so I thought of making it an addon.

![screen1](https://user-images.githubusercontent.com/1311555/107590266-9ffe2e80-6bff-11eb-83af-33c25ce3f0a8.png)
![screen2](https://zylannprods.fr/images/godot/plugins/atmosphere/screen2.png)

- Comes in two versions, one with fake colors and another with light scattering
- Can be seen from inside like regular fog if you want to land on the planet
- Switches to a cube mesh when seen from far away so multiple atmospheres can be drawn at lower cost
- Includes experimental volumetric clouds


How to use
-----------

Copy the contents of `zylann.atmosphere` under the `res://addons` folder of your project.
Activating the plugin in ProjectSettings is necessary to use `NoiseCubemap` resources.

A demo scene is available under `res://addons/zylann.atmosphere/demo`.

To start from scratch, drag and drop `planet_atmosphere.tscn` as child of your planet node. In the inspector, give it the same radius, and choose a height. You can tweak colors by expanding the `shader_params` category. Depending on the size of your planet, you may also have to tune density since light will have to travel larger distances through it.

### Shaders

The plugin has several shader variants in `res://addons/zylann.atmosphere/shaders`. They look and perform differently, and have different settings. You can choose one by assigning the `custom shader` field in the inspector.

- `planet_atmosphere_v1_no_clouds.gdshader`: original faked atmosphere
- `planet_atmosphere_v1_clouds.gdshader`: original faked atmosphere with volumetric clouds
- `planet_atmosphere_v1_clouds_high.gdshader`: original faked atmosphere with better-quality volumetric clouds, but more expensive
- `planet_atmosphere_no_clouds.gdshader`: atmosphere with scattering
- `planet_atmosphere_clouds.gdshader`: atmosphere with scattering and volumetric clouds
- `planet_atmosphere_clouds_high.gdshader`: atmosphere with scattering and better-quality volumetric clouds, but more expensive
- `planet_atmosphere_clouds_high_m.gdshader`: atmosphere with scattering with volumetric clouds using raymarched lighting, even more expensive
- `optical_depth.gdshader`: This shader is for internal use, it should not be assigned in the atmosphere node.

The atmosphere version with scattering was based on [Sebastian Lague's Coding Adventure](https://www.youtube.com/watch?v=dzcFB_9xHtg), with some help from [ProceduralPlanetGodot](https://github.com/athillion/ProceduralPlanetGodot).

### Clouds

![screen2](https://zylannprods.fr/images/godot/plugins/atmosphere/screen3.png)

Some shaders have volumetric clouds. They are quite expensive so tradeoffs had to be used. They are imperfect at the moment, but maybe in the future they can be made better as Godot gets more rendering APIs to help with post-processing effects.

Clouds need a coverage cubemap to work well, otherwise by default they cover the whole atmosphere uniformly. The plugin comes with a `NoiseCubemap` custom resource. It is similar to `NoiseTexture`, except it generates on a cubemap, which can be applied to a sphere seamlessly, without pinching at the poles.

### Known issues

- The effect will stop rendering when getting close to the planet in the editor. This is because the script can't access the camera of the main 3D viewport. It should work fine in game. To workaround this, enable `force_fullscreen`, and turn it back off once you're done tweaking.

- If your renderer is set to `Compatibility`, no atmosphere renders. It is currently unclear why. If you find out, PR is welcome.

- If you use Godot 4.2 or older, the atmosphere won't render properly. 4.3 broke compatibility when introducing reverse-Z depth, but didn't provide macros to test for it directly in the shader. So it was decided to bump support of the plugin from 4.3 onwards. To workaround that in older versions, comment out the `REVERSE_Z` macro inside `shaders/include/planet_atmosphere_main.gdshaderinc`.

- If you use a build of Godot with double-precision, [there is an inconsistency](https://github.com/godotengine/godot/pull/93108) with `INV_VIEW_MATRIX` which causes clouds to not render properly. To workaround this, enable the `DOUBLE_PRECISION` macro inside `shaders/include/planet_atmosphere_main.gdshaderinc`.
