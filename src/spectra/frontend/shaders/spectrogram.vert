#version 330 core

const vec2 VERTS[4] = vec2[](
    vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1)
);
const vec2 UVS[4] = vec2[](
    vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,1)
);

out vec2 uv;

void main() {
    uv = UVS[gl_VertexID];
    gl_Position = vec4(VERTS[gl_VertexID], 0, 1);
}
