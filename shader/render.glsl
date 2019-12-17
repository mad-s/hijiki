#version 450
#pragma shader_stage(compute)

layout(local_size_x=16, local_size_y=16) in;

#include "math.glsl"

#include "rand.glsl"

#include "quaternion.glsl"

struct Camera {
	vec4 position;
	vec4 rotation;
	float fov;
};


struct Ray {
	vec3 origin;
	vec3 direction;
	float tMin;
	float tMax;
};

Ray getCameraRayAt(Camera c, vec2 x, vec2 dimension) {
	x = x - 0.5 * dimension;
	x = x * tan(radians(0.5*c.fov)) / (0.5*dimension.x);

	Ray res;
	res.origin = c.position.xyz;
	res.direction = normalize(quaternionRotate(vec3(x.x, -x.y, -1.0), c.rotation));
	res.tMin = M_EPS;
	res.tMax = 1e100;
	return res;
}

// result of a ray-object intersection
struct Intersection {
	int objectID;
	float t;
	vec3 p;
	vec3 n;
	vec2 uv;
	mat3 frame;
};

struct ShapeQueryRecord {
	vec3 p;
	vec3 n;
	float pdf;
};


#include "materials/diffuse.glsl"
#include "materials/mirror.glsl"
#include "materials/dielectric.glsl"
#include "materials/emissive.glsl"
#include "materials/portal.glsl"

#include "material.glsl"

#include "shapes/sphere.glsl"
#include "shapes/quad.glsl"
#include "shapes/triangle.glsl"

#include "scene.glsl"

#include "block.glsl"



layout(set = 0, binding = 0) buffer CurrentImageBlock {
	ImageBlock currentImageBlock;
};

layout(RGBA32F, set=0, binding=1) uniform image2DArray outputImage;


void integrateRay(Ray ray, out vec3 total, out vec3 albedo, out float depth, out vec3 normal) {
	vec3 currentExtinction = vec3(0.);
	total = vec3(0.);
	bool hasAlbedo = false;
	albedo = vec3(0.);
	depth = 0;
	normal = vec3(0);

	vec3 throughput = vec3(1.);
	bool wasDiscrete = true;
	Intersection its;
	for (int bounce = 0; bounce < 1000; bounce++) {

		if (!intersectScene(ray, its)) {
			return;
		}
		//total = vec3(its.objectID);
		//return;

		if (bounce == 0) {
			depth  = its.t;
			normal = its.n;
		}

		uint mat = materials[its.objectID];
		uint material_tag = mat >> MATERIAL_TAG_SHIFT;
		uint material_idx = mat & ((1<<MATERIAL_TAG_SHIFT) - 1);

		float dist = distance(ray.origin, its.p);
		throughput *= exp(-currentExtinction * dist);

		if (material_tag == MATERIAL_TAG_EMISSIVE && wasDiscrete) {
			total += throughput * emissiveMaterials[material_idx].power;
		}
		if (material_tag == MATERIAL_TAG_DIFFUSE) {
			ShapeQueryRecord sRec;
			Ray shadowRay;
			vec3 importance = sampleEmitter(its.p, shadowRay);
			if (length(importance) > M_EPS && dot(shadowRay.direction, its.n) > 0) {
				if (!intersectScene(shadowRay)) {
					total += throughput * evalBSDF(mat, shadowRay.direction, its, -ray.direction) * importance;
				}
			}
			/*
			sampleQuad(quads[2], sRec);

			vec3 toLight = sRec.p - its.p;
			float r = length(toLight);
			toLight /= r;
			float cosTheta = dot(toLight, its.n);
			if (cosTheta > 0) {
				float cosThetaL = -dot(toLight, sRec.n);
				if (cosThetaL > 0) {
					float pdf = sRec.pdf * r*r / cosThetaL;

					Ray shadowRay;
					shadowRay.origin = its.p;
					shadowRay.direction = toLight;
					shadowRay.tMin = 2.*M_EPS;
					shadowRay.tMax = r-M_EPS;
					
					if (!intersectScene(shadowRay)) {
						total += throughput * evalBSDF(mat, toLight, its, -ray.direction) * emissiveMaterials[0].power / pdf;
					}
				}
			}
			*/
		}

		
		vec3 wo;
		throughput *= sampleBSDF(mat, ray.direction, its, wo, currentExtinction);
		ray.direction = wo;
		ray.origin = its.p;
		ray.tMin = 2.*M_EPS;
		ray.tMax = 1e100;

		wasDiscrete = material_tag != MATERIAL_TAG_DIFFUSE;

		if (bounce > 3) {
			float q = min(0.99, max(throughput.r, max(throughput.g, throughput.b)));
			if (randUniformFloat() > q) {
				break;
			} else {
				throughput /= q;
			}
		}
	}

}

void main() {
	uvec2 local  = uvec2(gl_GlobalInvocationID.xy);
	uvec2 global = local + currentImageBlock.origin;
	if (any(greaterThanEqual(local, currentImageBlock.originalDimension))) {
		return;
	}

	uint seed = currentImageBlock.seed + local.x + local.y*currentImageBlock.dimension.x;
	seedRng(seed);


	Ray ray = getCameraRayAt(camera,
		vec2(global) + currentImageBlock.sampleOffset,
		currentImageBlock.originalDimension);
	
	
	vec3 total;
	vec3 albedo;
	float depth;
	vec3 normal;
	integrateRay(ray, total, albedo, depth, normal);


	imageStore(outputImage, ivec3(local, 0), vec4(total, 1.));
	imageStore(outputImage, ivec3(local, 1), vec4(normal, depth));
	imageStore(outputImage, ivec3(local, 2), vec4(albedo, 0.));
}
