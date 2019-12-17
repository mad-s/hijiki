layout(set = 0, binding = BINDING_SCENE) buffer SceneBufferInfo {
	Camera camera;
	int numSpheres;
	int numQuads;
};

struct BVHNode {
	vec4 aabbMinShapeIndex;
	vec4 aabbMaxExitIndex;
};

layout(set = 0, binding = BINDING_BVH) buffer BVH {
	BVHNode bvh[];
};

layout(set = 0, binding = BINDING_SPHERES) buffer Spheres {
	Sphere spheres[];
};

layout(set = 0, binding = BINDING_QUADS) buffer Quads {
	Quad quads[];
};

layout(set = 0, binding = BINDING_MATERIALS) buffer Materials {
	uint materials[];
};

layout(RGBA32F, set=0, binding = BINDING_TEXTURE_IMAGES) uniform image2DArray imgTextures;

layout(RGBA32F, set=0, binding = BINDING_ENVMAP) uniform image2DArray envmap;

bool intersectScene(Ray ray, out Intersection its);
bool intersectScene(Ray ray) {
	// TODO: optimize
	Intersection dummy;
	return intersectScene(ray, dummy);
}

vec3 sampleEnvmap(Ray ray) {
        ivec2 size = imageSize(envmap).xy;
        return imageLoad(envmap, ivec3(sphericalToUvCoords(toSphericalCoords(ray.direction)) * size, 0)).xyz;
}

bool intersectScene(Ray ray, out Intersection its) {
	its.objectID = -1;
#if USE_BVH == 1
	vec3 invRayDir = 1.0 / ray.direction;
	vec3 timeOffset = -ray.origin * invRayDir;
	for(uint currentNode = 0; currentNode < bvh.length(); ) {
		uint shapeIndex = floatBitsToUint(bvh[currentNode].aabbMinShapeIndex.w);
		uint exitIndex  = floatBitsToUint(bvh[currentNode].aabbMaxExitIndex.w);
		if (shapeIndex != -1) {
			bool hasIntersection;
			if (shapeIndex < numSpheres) {
				hasIntersection = intersectSphere(ray, spheres[shapeIndex], its);
			} else {
				hasIntersection = intersectQuad(ray, quads[shapeIndex-numSpheres], its);
			}
			if (hasIntersection) {
				ray.tMax = its.t - M_EPS;
				its.objectID = int(shapeIndex);
			}
			currentNode = exitIndex;
		} else {
			vec3 tNegative = bvh[currentNode].aabbMinShapeIndex.xyz * invRayDir + timeOffset;
			vec3 tPositive = bvh[currentNode].aabbMaxExitIndex.xyz  * invRayDir + timeOffset;
			vec3 tMin = min(tNegative, tPositive);
			vec3 tMax = max(tNegative, tPositive);
			float t0 = max(max(tMin.x, tMin.y), tMin.z);
			float t1 = min(min(tMax.x, tMax.y), tMax.z);
			if (t0 < t1+M_EPS && t0 < ray.tMax && t1 > ray.tMin) {
				currentNode = currentNode+1; // continue in the tree
			} else {
				currentNode = exitIndex;
			}
		}
	}
#else
	if (numSpheres > 100 || numQuads > 100) {
		// failsafe
		return false;
	}

	for (int i = 0; i < numSpheres; i++) {
		if (intersectSphere(ray, spheres[i], its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = i;
		}
	}
	for (int i = 0; i < numQuads; i++) {
		if (intersectQuad(ray, quads[i], its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = numSpheres + i;
		}
	}
#endif
	
	if (its.objectID == -1) {
		return false;
	}

	its.p = ray.origin + its.t * ray.direction;
	// TODO: generalize
	if (its.objectID < numSpheres) {
		populateSphereIntersection(spheres[its.objectID], its);
	} else {
		populateQuadIntersection(quads[its.objectID-numSpheres], its);
	}

	return true;
}
