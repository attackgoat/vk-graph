const float PI = 3.14159265;

vec4 plasma(
    vec2 offset,
    vec2 extent,
    float time,
    float radius,
    float alpha,
    float size,
    vec3 shift
) {
    float color1 = (sin(dot(offset, vec2(sin(time * 3.0), cos(time * 3.0))) * 0.02 + time * 3.0) + 1.0) / 2.0;
    vec2 center = extent / 2.0 + vec2(extent.x / 2.0 * sin(-time * 3.0) * radius, extent.y / 2.0 * cos(-time * 3.0) * radius);
    float color2 = (cos(length(offset - center) * size) + 1.0) / 2.0;
    float color = (color1 + color2) / 2.0;

    float red = (sin(PI * color / shift.r + time * 3.0) + 1.0) / 2.0;
    float green = (sin(PI * color / shift.g + time * 3.0) + 1.0) / 2.0;
    float blue = (sin(PI * color / shift.b + time * 3.0) + 1.0) / 2.0;

    return vec4(red, green, blue, alpha);
}
