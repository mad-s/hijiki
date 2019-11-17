uint rngState;
uint randUint() {
	rngState ^= (rngState << 13);
	rngState ^= (rngState >> 17);
	rngState ^= (rngState << 5);
	return rngState;
}

void seedRng(uint seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	rngState = seed;
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

vec3 randUniformSphere() {
	float u = randUniformFloat();
	float v = randUniformFloat();

	float z = 2.*u-1.;
	float theta = 2*M_PI*v;
	float r = sqrt(1-z*z);
	return vec3(r*cos(theta), r*sin(theta), z);
}

