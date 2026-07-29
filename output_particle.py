# Python unpacking function for output_particle array:
# 
# Given an [N,6] uint32 array from a little-endian machine, unpack into
# position[N,3], velocity[N,3], pid[N,3], density[N], velocity_dispersion[N],
# is_map[N], multiplicity[N], relative_velocity[N].
# 
# MAPs are identified by pid_k == 0xffff. For MAPs, the multiplicity and
# relative_velocity fields are meaningful; for DM particles they are zero.
# The pid field for MAPs contains undefined values.
# 
# Usage:
#    pos, vel, pid, density, vel_disp, is_map, mult, rel_vel = unpack_output_particles(data)
# 
# where data is the [N,6] uint32 array.

import numpy as np

def unpack_output_particles(data):
    """
    Unpack output_particle data from [N,6] uint32 array.

    Handles both DM particles and MAPs. MAPs are identified by pid_k == 0xffff.

    Returns:
        pos: [N,3] float32 array in Mpc/h
        vel: [N,3] float32 array in km/s
        pid: [N,3] uint16 array (Lagrangian coordinates for DM; undefined for MAPs)
        density: [N] float32 array
        velocity_dispersion: [N] float32 array
        is_map: [N] bool array (True for MAPs, False for DM)
        multiplicity: [N] uint32 array (MAP multiplicity; 0 for DM particles)
        relative_velocity: [N] float32 array (MAP relative velocity; 0 for DM)
    """
    data = np.ascontiguousarray(data).view(np.uint32)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Expected [N,6] uint32 or [N,24] uint8; got shape {data.shape} after reinterpretation")
    N = data.shape[0]

    # Position: uint32[0:3] are int32, multiply by 1/131072 to get Mpc/h
    pos = data[:, 0:3].copy().view(np.int32).astype(np.float32) * (1.0/131072.0)

    # Velocity: uint32[3] contains packed vect32
    # Unpack the vect32 format:
    # Byte 0: sign bits (bit 7=vx, bit 6=vy, bit 5=vz) and 5-bit exponent
    # Bytes 1,2,3: mantissas for vx,vy,vz
    vel_packed = data[:, 3]
    vel_bytes = vel_packed.copy().view(np.uint8).reshape(N, 4)

    # Extract sign bits (top 3 bits of byte 0)
    signs = vel_bytes[:, 0:1] >> np.array([7, 6, 5], dtype=np.uint8)
    signs = (signs & 1).astype(np.float32) * np.float32(-2.0) + np.float32(1.0)

    # Extract exponent (bottom 5 bits of byte 0)
    exponent = vel_bytes[:, 0] & 0x1F

    # Check for overflow (exponent == 31)
    overflow = (exponent == 31)

    # Extract mantissas (bytes 1,2,3)
    mantissas = vel_bytes[:, 1:4].astype(np.float32)

    # Compute velocities: mantissa * 2^exponent
    # But need to limit exponent to 23 to avoid int32 overflow
    shift = np.minimum(exponent, 23)[:, np.newaxis]
    vel = (signs * mantissas * (2.0 ** shift)).astype(np.float32)
    vel[overflow] = np.nan

    # Check if particle is a MAP: pid_k == 0xffff
    pid_k = data[:, 5] & 0xFFFF
    is_map = (pid_k == 0xFFFF)

    # PID: uint32[4] contains pid[0] and pid[1], uint32[5] lower 16 bits contains pid[2]
    # For MAPs, this is undefined/nonsense
    pid = np.empty((N, 3), dtype=np.uint16)
    pid[:, 0] = data[:, 4] & 0xFFFF
    pid[:, 1] = (data[:, 4] >> 16) & 0xFFFF
    pid[:, 2] = pid_k

    # MAP-specific data: uint32[4] contains multiplicity (upper 24 bits) and relative velocity (lower 8 bits)
    map_data = data[:, 4]
    multiplicity = np.zeros(N, dtype=np.uint32)
    multiplicity[is_map] = (map_data[is_map] >> 8)

    relative_velocity_byte = (map_data & 0xFF).astype(np.uint8)
    relative_velocity = np.zeros(N, dtype=np.float32)
    relative_velocity[is_map] = unpack_ufloat8_35(relative_velocity_byte[is_map])

    # Density: byte 2 of uint32[5] (bits 16-23), ufloat8_44 format
    density_byte = ((data[:, 5] >> 16) & 0xFF).astype(np.uint8)
    density = unpack_ufloat8_44(density_byte)

    # Velocity dispersion: byte 3 of uint32[5] (bits 24-31), ufloat8_35 format
    vel_disp_byte = ((data[:, 5] >> 24) & 0xFF).astype(np.uint8)
    velocity_dispersion = unpack_ufloat8_35(vel_disp_byte)

    return pos, vel, pid, density, velocity_dispersion, is_map, multiplicity, relative_velocity


_recarray_dtype = np.dtype([
    ('pos',      np.float32, (3,)),
    ('vel',      np.float32, (3,)),
    ('pid',      np.uint16,  (3,)),
    ('density',  np.float32),
    ('vel_disp', np.float32),
    ('is_map',   np.bool_),
    ('mult',     np.uint32),
    ('rel_vel',  np.float32),
])

def unpack_output_particles_recarray(data):
    """
    Like unpack_output_particles, but returns a structured recarray with fields:
    pos, vel, pid, density, vel_disp, is_map, mult, rel_vel.
    pos and vel are 3-vectors; all floating-point fields are float32.
    """
    pos, vel, pid, density, vel_disp, is_map, mult, rel_vel = unpack_output_particles(data)
    result = np.empty(pos.shape[0], dtype=_recarray_dtype).view(np.recarray)
    result.pos      = pos
    result.vel      = vel
    result.pid      = pid
    result.density  = density
    result.vel_disp = vel_disp
    result.is_map   = is_map
    result.mult     = mult
    result.rel_vel  = rel_vel
    return result


def unpack_ufloat8_44(values):
    """
    Unpack ufloat8_44: 4 bits exponent, 4 bits mantissa.
    The encoding represents subnormal floats with bottom 8 bits.
    Magic number for unpacking: 2^-130 (bottom 8 bits of float32).
    """
    # values is uint8 array
    # Unpack by treating as bottom 8 bits of float32 mantissa and multiplying
    # Shift to form a float32 with the value in the mantissa position
    u = values.astype(np.uint32) << 15  # Put in float32 mantissa position
    f = u.view(np.float32)
    # Magic number is 2^130 (since we encoded with 2^-130)
    magic = np.float64(2.0**130)
    return (f.astype(np.float64) * magic).astype(np.float32)


def unpack_ufloat8_35(values):
    """
    Unpack ufloat8_35: 3 bits exponent, 5 bits mantissa.
    Magic number for unpacking: 2^-131.
    """
    u = values.astype(np.uint32) << 15
    f = u.view(np.float32)
    magic = np.float64(2.0**131)
    return (f.astype(np.float64) * magic).astype(np.float32)
