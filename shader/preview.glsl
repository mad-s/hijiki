#version 450
#pragma shader_stage(fragment)

layout(RGBA32F, set=0, binding=0) uniform image2D image;

layout(origin_upper_left) in vec4 gl_FragCoord;
layout(location=0) out vec4 outColor;

void main() {
	vec4 color = imageLoad(image, ivec2(gl_FragCoord.xy));
	outColor = vec4(color.rgb / color.w, 1.0);
}
