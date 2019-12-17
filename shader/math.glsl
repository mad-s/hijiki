#define M_PI 3.1415926535897932384626433832795
#define M_EPS 1e-4

vec2 toSphericalCoords(vec3 direction) {
        float theta = acos(direction.y);
        float phi = atan(direction.z, direction.x) + M_PI;
        return vec2(phi, theta);
}

vec3 toCartesianCoords(vec2 direction) {
        float x = sin(direction.y) * cos(direction.x);
        float y = sin(direction.y) * sin(direction.x);
        float z = cos(direction.y);
        return vec3(x, y, z);
}

vec2 sphericalToUvCoords(vec2 spherical) {
        return vec2(spherical.x / 2. / M_PI, spherical.y / M_PI);
}

vec2 uvToSphericalCoords(vec2 uvCoords) {
        return vec2(uvCoords.x * 2 * M_PI, uvCoords.y * M_PI);
}
