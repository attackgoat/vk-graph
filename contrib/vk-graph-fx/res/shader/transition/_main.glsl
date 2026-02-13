void main() {
    vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) / vec2(imageSize(dest_image));
    vec4 color = transition(uv);

    imageStore(dest_image, ivec2(gl_GlobalInvocationID.xy), color);
}
