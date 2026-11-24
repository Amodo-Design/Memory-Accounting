/* VDE field arithmetic for p = 2^89 - 1. The three limbs are radix 2^30,
 * with 29 bits in the most significant limb. These operations also compile
 * as plain C so that the verifier and GPU use identical representations. */
#ifndef POSM_FIELD89_H
#define POSM_FIELD89_H
#define F89_MASK30 ((1u << 30) - 1)
#define F89_MASK29 ((1u << 29) - 1)
typedef struct { u32 a, b, c; } field89;

INLINE field89 f89_reduce(u64 a, u64 b, u64 c) {
    b += a >> 30; a &= F89_MASK30;
    c += b >> 30; b &= F89_MASK30;
    /* 2^89 is 1. A product needs one wide fold and one final carry. */
    a += c >> 29; c &= F89_MASK29;
    b += a >> 30; a &= F89_MASK30;
    c += b >> 30; b &= F89_MASK30;
    a += c >> 29; c &= F89_MASK29;
    b += a >> 30; a &= F89_MASK30;
    c += b >> 30; b &= F89_MASK30;
    if (a == F89_MASK30 && b == F89_MASK30 && c == F89_MASK29)
        a = b = c = 0;
    field89 out = {(u32)a, (u32)b, (u32)c};
    return out;
}
INLINE field89 f89_add(field89 x, field89 y) {
    return f89_reduce((u64)x.a+y.a, (u64)x.b+y.b, (u64)x.c+y.c);
}
INLINE field89 f89_neg(field89 x) {
    return f89_reduce(F89_MASK30-x.a, F89_MASK30-x.b, F89_MASK29-x.c);
}
INLINE field89 f89_square(field89 x) {
    /* In radix 2^30 the fourth limb wraps with factor 2 (2^90 = 2 mod p).
     * Each unreduced accumulation is smaller than 2^63. */
    u64 a=x.a, b=x.b, c=x.c;
    return f89_reduce(a*a + 4*b*c, 2*a*b + 2*c*c, b*b + 2*a*c);
}
INLINE field89 f89_encode(field89 d, field89 tweak, u32 *nonresidue) {
    field89 x=f89_add(d,tweak), root=x;
    for (int i=0;i<87;i++) root=f89_square(root);
    field89 square=f89_square(root);
    *nonresidue=(square.a!=x.a || square.b!=x.b || square.c!=x.c);
    return root;
}
INLINE field89 f89_decode(field89 root, u32 nonresidue, field89 tweak) {
    field89 square=f89_square(root);
    return f89_add(nonresidue ? f89_neg(square) : square, f89_neg(tweak));
}
#endif
