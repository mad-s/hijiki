#version 450
#pragma shader_stage(compute)

//layout(local_size_x=16,local_size_y=16) in;

#include "math.glsl"
#include "block.glsl"
#include "rand.glsl"

layout(set=0, binding=0) buffer CurrentImageBlock {
	ImageBlock currentImageBlock;
};

layout(set=0, binding=1) buffer IntegratorOutputs {
	vec4 integratorOutputs[];
};

layout(RGBA32F, set=0, binding=1) uniform image2D inputImage;
layout(RGBA32F, set=0, binding=2) uniform image2D outputImage;


void main() {
	uvec2 local = gl_GlobalInvocationID.xy-RECONSTRUCTION_RADIUS;
	uvec2 global = local + currentImageBlock.origin;
	vec4 outputValue = imageLoad(outputImage, ivec2(global));
	// TODO
	float gaussFac = -1. / (2*RECONSTRUCTION_STDDEV*RECONSTRUCTION_STDDEV);
	float curveOffset = exp(gaussFac * RECONSTRUCTION_RADIUS*RECONSTRUCTION_RADIUS);
	for (int dx = -RECONSTRUCTION_RADIUS; dx <= RECONSTRUCTION_RADIUS; dx++) {
		for (int dy = -RECONSTRUCTION_RADIUS; dy <= RECONSTRUCTION_RADIUS; dy++) {
			ivec2 offs = ivec2(dx, dy);

			vec2 sampleOffset = offs + currentImageBlock.sampleOffset - 0.5;
			float weight = exp(gaussFac*dot(sampleOffset, sampleOffset))-curveOffset;
			outputValue += weight * imageLoad(inputImage, ivec2(local+offs));
		}
	}
	imageStore(outputImage, ivec2(global), outputValue);
}
