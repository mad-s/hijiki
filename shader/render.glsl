#version 450
#pragma shader_stage(compute)

#define M_PI 3.1415926535897932384626433832795
#define M_EPS 1e-5

#include "rand.glsl"

#include "quaternion.glsl"

struct Camera {
	vec4 position;
	float fov;
};

struct Ray {
	vec3 origin;
	vec3 direction;
	float tMin;
	float tMax;
};

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


layout(set = 0, binding = 0) buffer Input {
	vec4 inputDirection[];
};

layout(set = 0, binding = 1) buffer Output {
	vec4 outputColor[];
};

struct SampleInfo {
	uint index;
	float weight;
	vec2 sampleOffset;
};
layout(set = 0, binding = 2) buffer CurrentSampleInfo {
	SampleInfo currentSampleInfo;
};



void main() {
	uint index = gl_GlobalInvocationID.x;
	seedRng(index*65536 + currentSampleInfo.index);
	if (currentSampleInfo.index == 0) {
		outputColor[index] = vec4(0.);
	}

	Ray ray;
	ray.origin = camera.position.xyz;
	ray.direction = normalize(inputDirection[index].xyz+vec3(currentSampleInfo.sampleOffset, 0.));
	ray.tMin = M_EPS;
	ray.tMax = 1e100;
	Intersection its;


	vec3 total = vec3(0.);
	vec3 throughput = vec3(1.);
	bool wasDiscrete = true;
	for (int bounce = 0; bounce < 100; bounce++) {
		if (!intersectScene(ray, its)) {
			break;
		}


		uint mat = materials[its.objectID];

		//outputColor[index] = vec4(its.objectID, mat, 0., 0.);
		//return;
		
		bool wasPortal = false;
		if (mat < numDiffuse) {
			// sample emitter
			// TODO: true and flexible MIS
			
			{
				ShapeQueryRecord sRec;
				//sampleSphere(spheres[2], sRec);
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
							//outputColor[index] = vec4(diffuseMaterials[mat].color * emissiveMaterials[0].power, 0.);
							//outputColor[index] = vec4(16 * throughput * diffuseMaterials[mat].color / M_PI * cosTheta * emissiveMaterials[0].power / pdf, 0.);
							//return;
						}
					}
				}
				//outputColor[index] = vec4(0.,0.,0., 0.);
				//return;
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

	outputColor[index] += currentSampleInfo.weight * vec4(total, 0.);
}
