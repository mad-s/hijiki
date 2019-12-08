#version 450
#pragma shader_stage(compute)

layout(local_size_x=16,local_size_y=16) in;

#include "math.glsl"
#include "block.glsl"
#include "rand.glsl"

layout(set=0, binding=0) buffer CurrentImageBlock {
	ImageBlock currentImageBlock;
};

layout(set=0, binding=1) buffer IntegratorOutputs {
	vec4 integratorOutputs[];
};

layout(RGBA32F, set=0, binding=1) uniform image2DArray inputImage;
layout(RGBA32F, set=0, binding=2) uniform image2D outputImage;


void main() {
	uvec2 local = gl_GlobalInvocationID.xy-RECONSTRUCTION_RADIUS;
	//uvec2 local = gl_GlobalInvocationID.xy;
	uvec2 global = local + currentImageBlock.origin;
	vec4 outputValue = imageLoad(outputImage, ivec2(global));

	// TODO: optimize loop indices
	float gaussFac = -1. / (2*RECONSTRUCTION_STDDEV*RECONSTRUCTION_STDDEV);
	float curveOffset = exp(gaussFac * RECONSTRUCTION_RADIUS*RECONSTRUCTION_RADIUS);

	vec3 normalCenter = imageLoad(inputImage, ivec3(local, 1)).xyz;
	vec3 albedoCenter = imageLoad(inputImage, ivec3(local, 2)).rgb;

	for (int dx = -RECONSTRUCTION_RADIUS; dx <= RECONSTRUCTION_RADIUS; dx++) {
		//if (local.x + dx < 0 || local.x + dx >= currentImageBlock.dimension.x)
		//	continue;
		for (int dy = -RECONSTRUCTION_RADIUS; dy <= RECONSTRUCTION_RADIUS; dy++) {
			//if (local.y + dy < 0 || local.y + dy >= currentImageBlock.dimension.y)
			//	continue;
			ivec2 offs = ivec2(dx, dy);

			vec2 sampleOffset = offs + currentImageBlock.sampleOffset - 0.5;
			float weight = exp(gaussFac*dot(sampleOffset, sampleOffset))-curveOffset;
			if (weight < 0)
				continue;
			vec4 color_weight = imageLoad(inputImage, ivec3(local+offs, 0));
			vec4 normal_depth = imageLoad(inputImage, ivec3(local+offs, 1));
			vec4 albedo       = imageLoad(inputImage, ivec3(local+offs, 2));


			vec3 normalOffset = normal_depth.xyz - normalCenter;
			vec3 albedoOffset = albedo.rgb       - albedoCenter;
			weight *= exp(-(dot(normalOffset, normalOffset)*2+dot(albedoOffset,albedoOffset)));
			outputValue += weight * color_weight;
		}
	}

	//vec4 color_weight = imageLoad(inputImage, ivec3(local, 0));
	//outputValue += color_weight;
	imageStore(outputImage, ivec2(global), outputValue);
}
