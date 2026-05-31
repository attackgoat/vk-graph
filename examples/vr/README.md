<image alt="Preview" src="../../.github/img/vr.png" width=40%>

# Virtual Reality Example

_Exaimine the skeleton of a Woolly Mammoth using the hands of Abraham Lincoln_

This example uses assets from the Smithsonian collection to create a real-life-size display in VR.
Any OpenXR-supported VR headset should work.
The example defaults to the runtime OpenXR loader (`loaded`); enable the `linked` feature if you want to
link against a system loader instead.

## Features

- Asset loading
- Basic VR handling/driver code
- Normal mapping using Mikktspace
- Hot shader reloading

## Tested Platforms

- Windows/Valve Index on SteamVR @ 144hz
