#version 450
#pragma shader_stage(vertex)

out gl_PerVertex {
    vec4 gl_Position;
};

const vec2 positions[4] = vec2[4](
    vec2(-1, 1),
    vec2(-1,-1),
    vec2( 1, 1),
    vec2( 1,-1)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
