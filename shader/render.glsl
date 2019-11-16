#version 450
#pragma shader_stage(compute)

#define M_PI 3.1415926535897932384626433832795
#define M_EPS 1e-5

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

struct Camera {
	vec3 position;
	float fov;
};
layout(set = 0, binding = 3) buffer SceneBufferInfo {
	Camera camera;
	int numSpheres;
	int numPlanes;

	int numDiffuse;
	int numMirrors;
	int numDielectric;
	int numEmitters;
};

struct Sphere {
	vec4 positionRadius;
};
layout(set = 0, binding = 4) buffer Spheres {
	Sphere spheres[];
};

struct Plane {
	vec4 normalOffset;
};
layout(set = 0, binding = 5) buffer Planes {
	Plane planes[];
};

layout(set = 0, binding = 6) buffer Materials {
	uint materials[];
};

struct DiffuseMaterial {
	vec3 color;
};
layout(set = 0, binding = 7) buffer DiffuseMaterials {
	 DiffuseMaterial diffuseMaterials;
};

struct MirrorMaterial {
	vec4 dummy;
};
layout(set = 0, binding = 8) buffer MirrorMaterials {
	 MirrorMaterial mirrorMaterials;
};

struct DielectricMaterial {
	float etaRatio;
	float pad[3];
};
layout(set = 0, binding = 9) buffer DielectricMaterials {
	 DielectricMaterial dielectricMaterials;
};

struct EmissiveMaterial {
	vec3 power;
};
layout(set = 0, binding = 10) buffer EmissiveMaterials {
	 EmissiveMaterial emissiveMaterials;
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
	vec3 pos;
	vec3 normal;
	vec2 uv;
};




bool intersectSphere(Ray ray, Sphere sphere, inout Intersection its);
bool intersectPlane(Ray ray, Plane plane, inout Intersection its);

// TODO: use BVH
bool intersectScene(Ray ray, out Intersection its) {
	its.objectID = -1;
	for (int i = 0; i < numSpheres; i++) {
		if (intersectSphere(ray, spheres[i], its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = i;
		}
	}
	for (int i = 0; i < numPlanes; i++) {
		if (intersectPlane(ray, planes[i], its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = numSpheres + i;
		}
	}
	if (its.objectID == -1) {
		return false;
	}

	its.pos = ray.origin + its.t * ray.direction;
	if (its.objectID < numSpheres) {
		its.normal = (its.pos - spheres[its.objectID].positionRadius.xyz) / spheres[its.objectID].positionRadius.w;
	} else {
		its.normal = planes[its.objectID-numSpheres].normalOffset.xyz;
	}

	return true;
}

bool intersectSphere(Ray ray, Sphere sphere, inout Intersection its) {
	vec3 l = ray.origin - sphere.positionRadius.xyz;
	float b = 2 * dot(ray.direction, l);
	float c = dot(l, l) - sphere.positionRadius.r * sphere.positionRadius.r;
	float d = b*b-4*c;
	if (d < M_EPS) {
		return false;
	}
	d = sqrt(d);
	float t0 = -0.5*(b+d);
	if (ray.tMin <= t0 && t0 <= ray.tMax) {
		its.t = t0;
		return true;
	}
	float t1 = -0.5*(b-d);
	if (ray.tMin <= t0 && t0 <= ray.tMax) {
		its.t = t1;
		return true;
	}
	return false;
}

bool intersectPlane(Ray ray, Plane plane, inout Intersection its) {
	vec3 n  = plane.normalOffset.xyz;
	float a = plane.normalOffset.w;
	float t = -(dot(ray.origin, n)+a) / dot(ray.direction, n);
	if (ray.tMin <= t && t <= ray.tMax) {
		its.t = t;
		return true;
	}
	return false;
}

uint rngState;
uint randUint() {
	rngState ^= (rngState << 13);
	rngState ^= (rngState >> 17);
	rngState ^= (rngState << 5);
	return rngState;
}

uint wangHash(uint seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}

float randUniformFloat() {
	return float(randUint()) * (1.0 / 4294967296.0);
}

vec3 randCosHemisphere() {
	float u = randUniformFloat();
	float v = randUniformFloat();
	float r = sqrt(u);
	float theta = 2*M_PI*v;
	float x = r*cos(theta);
	float y = r*sin(theta);
	return vec3(x, y, sqrt(max(0, 1-u)));
}

vec4 quaternionMult(vec4 qa, vec4 qb) {
	vec4 result;
	result.w = qa.w*qb.w - dot(qa.xyz, qb.xyz);
	result.xyz = cross(qa.xyz, qb.xyz) + qa.xyz * qb.w + qb.xyz * qa.w;
	return result;
}

vec4 quaternionFromTo(vec3 a, vec3 b) {
	vec4 result;
	result.xyz = cross(a, b);
	result.w = 1+dot(a, b);
	return normalize(result);
}

vec3 quaternionRotate(vec3 v, vec4 r) {
	vec4 tmp = quaternionMult(r, vec4(v, 0.));
	r.xyz = -r.xyz;
	return quaternionMult(tmp, r).xyz;
}

void main() {
	uint index = gl_GlobalInvocationID.x;
	rngState = wangHash(index*65536 + currentSampleInfo.index);
	if (currentSampleInfo.index == 0) {
		outputColor[index] = vec4(0.);
	}

	vec3 total = vec3(0.);
	Ray ray;
	ray.origin = camera.position;
	ray.direction = normalize(inputDirection[index].xyz+vec3(currentSampleInfo.sampleOffset, 0.));
	ray.tMin = M_EPS;
	ray.tMax = 1e100;

	Intersection its;

	!intersectScene(ray, its);
	outputColor[index] = vec4(its.objectID);
}
