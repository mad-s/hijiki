layout(set = 0, binding = 2) buffer SceneBufferInfo {
	Camera camera;
	int numSpheres;
	int numPlanes;
	int numQuads;

	int numDiffuse;
	int numMirrors;
	int numDielectric;
	int numEmitters;
	int numPortals;
};

layout(set = 0, binding = 3) buffer Spheres {
	Sphere spheres[];
};

layout(set = 0, binding = 4) buffer Planes {
	Plane planes[];
};
layout(set = 0, binding = 5) buffer Quads {
	Quad quads[];
};

layout(set = 0, binding = 6) buffer Materials {
	uint materials[];
};

// TODO: use BVH
bool intersectScene(Ray ray, out Intersection its);
bool intersectScene(Ray ray) {
	// TODO: optimize
	Intersection dummy;
	return intersectScene(ray, dummy);
}
bool intersectScene(Ray ray, out Intersection its) {
	its.objectID = -1;
	if (numSpheres > 100 || numPlanes > 100) {
		// failsafe
		return false;
	}

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
	for (int i = 0; i < numQuads; i++) {
		if (intersectQuad(ray, quads[i], its)) {
			ray.tMax = its.t - M_EPS;
			its.objectID = numSpheres + numPlanes + i;
		}
	}
	
	if (its.objectID == -1) {
		return false;
	}

	its.p = ray.origin + its.t * ray.direction;
	// TODO: generalize
	if (its.objectID < numSpheres) {
		its.n = (its.p - spheres[its.objectID].positionRadius.xyz) / spheres[its.objectID].positionRadius.w;
	} else {
		if (its.objectID < numSpheres + numPlanes) {
			its.n = planes[its.objectID-numSpheres].normalOffset.xyz;
		} else {
			Quad quad = quads[its.objectID-numSpheres-numPlanes];
			its.n = normalize(cross(quad.edge1,quad.edge2));
		}
	}

	return true;
}
