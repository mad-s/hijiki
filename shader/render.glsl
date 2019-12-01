#version 450
#pragma shader_stage(compute)

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
	x = x * tan(radians(0.5*c.fov)) / (0.5*dimension.y);

	Ray res;
	res.origin = c.position.xyz;
	res.direction = normalize(vec3(x.x, -x.y, -1.0)); // TODO: camera rotation
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
};

struct ShapeQueryRecord {
	vec3 p;
	vec3 n;
	float pdf;
};

#include "shapes/sphere.glsl"
#include "shapes/plane.glsl"
#include "shapes/quad.glsl"

#include "scene.glsl"

#include "materials/diffuse.glsl"
#include "materials/mirror.glsl"
#include "materials/dielectric.glsl"
#include "materials/emissive.glsl"
#include "materials/portal.glsl"

#include "material.glsl"

#include "block.glsl"


layout(set = 0, binding = 0) buffer CurrentImageBlock {
	ImageBlock currentImageBlock;
};

layout(RGBA32F, set=0, binding=1) uniform image2DArray outputImage;

void main() {
	uvec2 local  = uvec2(gl_GlobalInvocationID.xy);
	uvec2 global = local + currentImageBlock.origin;

	uint seed = currentImageBlock.seed + local.x + local.y*currentImageBlock.dimension.x;
	seedRng(seed);


	Ray ray = getCameraRayAt(camera,
		vec2(global) + currentImageBlock.sampleOffset,
		currentImageBlock.originalDimension);
	Intersection its;


	vec3 total = vec3(0.);
	bool hasAlbedo = false;
	vec3 albedo = vec3(0.);
	float depth = 0;
	vec3 normal = vec3(0);
	vec3 throughput = vec3(1.);
	bool wasDiscrete = true;
	for (int bounce = 0; bounce < 100; bounce++) {

		if (!intersectScene(ray, its)) {
			break;
		}
		if (bounce == 0) {
			depth  = its.t;
			normal = its.n;
		}

		uint mat = materials[its.objectID];

		bool wasPortal = false;
		if (mat < numDiffuse) {
			if (!hasAlbedo) {
				albedo = diffuseMaterials[mat].color;
				hasAlbedo = true;
			}
			// sample emitter
			// TODO: true and flexible MIS
			
			{
				ShapeQueryRecord sRec;
				sampleQuad(quads[0], sRec);


				vec3 toLight = sRec.p - its.p;
				float r = length(toLight);
				toLight /= r;
				float cosTheta = dot(toLight, its.n);
				if (cosTheta > 0) {
					float cosThetaL = -dot(toLight, sRec.n);
					if (cosThetaL > 0) {
						float pdf = sRec.pdf * r*r / cosThetaL;

						Ray shadowRay;
						ray.origin = its.p;
						ray.direction = toLight;
						ray.tMin = M_EPS;
						ray.tMax = r-M_EPS;
						
						if (!intersectScene(ray)) {
							total += throughput * diffuseMaterials[mat].color / M_PI * cosTheta * emissiveMaterials[0].power / pdf;
						}
					}
				}
			}
			wasDiscrete = false;
			vec3 wo = randCosHemisphere();
			vec4 localToWorld = quaternionFromTo(vec3(0.,0.,1), its.n);
			ray.direction = quaternionRotate(wo, localToWorld);

			throughput *= diffuseMaterials[mat].color;
		} else {
			mat -= numDiffuse;
			if (mat < numMirrors) {
				ray.direction = reflect(ray.direction, its.n);
			} else {
				mat -= numMirrors;
				if (mat < numDielectric) {
					// TODO: optimize
					float eta = dielectricMaterials[mat].etaRatio;
					float etaInv = 1. / eta;
					float cosThetaI = -dot(its.n, ray.direction);
					vec3 normal = its.n;
					if (cosThetaI < 0) {
						eta = etaInv;
						etaInv = 1. / eta;
						normal = -normal;
						cosThetaI = -cosThetaI;
					}

					float k = 1.0 - etaInv*etaInv * (1-cosThetaI*cosThetaI);

					if (k <= 0) {
						// reflect
						ray.direction = reflect(ray.direction, normal);
					} else {
						float cosThetaO = sqrt(k);

						float rho_par  = (eta*cosThetaI-cosThetaO)/(eta*cosThetaI+cosThetaO);
						float rho_orth = (cosThetaI-eta*cosThetaO)/(cosThetaI+eta*cosThetaO);

						float f_r = 0.5 * (rho_par*rho_par + rho_orth*rho_orth);
						if (randUniformFloat() < f_r) {
							ray.direction = reflect(ray.direction, normal);
						} else {
							vec3 parallel = ray.direction - dot(ray.direction, normal) * normal;
							// refract
							ray.direction = etaInv * parallel - sqrt(k) * normal;
						}
						
					}
				} else {
					mat -= numDielectric;
					if (mat < numEmitters) {
						// emitters
						if (wasDiscrete) {
							total += throughput * emissiveMaterials[mat].power;
						}
						break;
					} else {
						mat -= numEmitters;
						ray.direction = normalize((portalMaterials[mat].transform * vec4(ray.direction, 0)).xyz);
						ray.origin    = (portalMaterials[mat].transform * vec4(its.p, 1)).xyz;
						wasPortal = true;
					}
				}
			}
			wasDiscrete = true;
		}

		if (!wasPortal) {
			ray.origin = its.p;
		}
		ray.tMax = 1e100;

		float q = min(0.9, max(throughput.r, max(throughput.g, throughput.b)));
		if (randUniformFloat() > q) {
			break;
		} else {
			throughput /= q;
		}
	}

	imageStore(outputImage, ivec3(local, 0), vec4(total, 1.));
	imageStore(outputImage, ivec3(local, 1), vec4(normal, depth));
	imageStore(outputImage, ivec3(local, 2), vec4(albedo, 0.));
	// TODO: normal
}
