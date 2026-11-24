#ifndef POSM_CODEC89_H
#define POSM_CODEC89_H
#include "field89.h"

/* Five 12-byte slots and 4 zero bytes per 64 encoded bytes. Each slot
 * represents 11 plaintext bytes. Eight rows align plaintext to uint64. */
INLINE field89 f89_from_words(u32 a,u32 b,u32 c) {
    field89 x={a & F89_MASK30, ((a>>30)|(b<<2)) & F89_MASK30, (b>>28)|(c<<4)};
    return x;
}
INLINE field89 f89_key(struct u128 hash) {
    field89 x=f89_from_words((u32)hash.lo,(u32)(hash.lo>>32),(u32)hash.hi & 0x1ffffffu);
    return f89_reduce(x.a,x.b,x.c);
}
INLINE field89 f89_slot_tweak(u64 kl,u64 kh,u64 q,u32 lane,field89 base) {
#if POSM_GROUP64
    field89 offset={lane,0,0};
    return f89_add(base,offset);
#else
    return f89_key(siphash24_128(kl,kh,q+lane,TWEAK_DOMAIN,0,0));
#endif
}
INLINE field89 f89_base(u64 kl,u64 kh,u64 q) {
#if POSM_GROUP64
    return f89_key(siphash24_128(kl,kh,q,TWEAK_DOMAIN,0,0));
#else
    field89 zero={0,0,0}; return zero;
#endif
}
INLINE field89 f89_read_plain(const u8 *raw,u64 available,u64 offset) {
    u32 words[3]={0,0,0};
    for (u32 i=0;i<11;i++) if (offset+i<available)
        words[i/4] |= (u32)raw[offset+i] << (8*(i%4));
    return f89_from_words(words[0],words[1],words[2]);
}
INLINE void f89_plain_bytes(u8 *raw,u64 available,u64 offset,field89 x) {
    u32 words[3]={x.a|(x.b<<30),(x.b>>2)|(x.c<<28),x.c>>4};
    for (u32 i=0;i<11;i++) if (offset+i<available)
        raw[offset+i]=(u8)(words[i/4]>>(8*(i%4)));
}
INLINE void f89_write_root(u32 *words,field89 d,field89 t) {
    u32 flag; field89 r=f89_encode(d,t,&flag);
    words[0]=r.a|(r.b<<30); words[1]=(r.b>>2)|(r.c<<28);
    words[2]=(r.c>>4)|(flag<<25);
}
INLINE field89 f89_read_root(const u32 *w,field89 t) {
    return f89_decode(f89_from_words(w[0],w[1],w[2]&0x1ffffffu),(w[2]>>25)&1,t);
}
/* Setup reads an already slot-packed payload. KV uses the continuous raw stream. */
INLINE void f89_setup_row(const u64 *raw,u32 *out,u64 kl,u64 kh,u64 q) {
    field89 base=f89_base(kl,kh,q);
    for (u32 lane=0;lane<5;lane++) {
        field89 d=f89_read_plain((const u8*)raw,64,12*lane);
        f89_write_root(out+3*lane,d,f89_slot_tweak(kl,kh,q,lane,base));
    }
    out[15]=0;
}
INLINE void f89_transform_row(u8 *raw,u64 available,u64 offset,u64 *encoded,
                              u64 q,u64 kl,u64 kh,int decode) {
    field89 base=f89_base(kl,kh,q);
    for (u32 lane=0;lane<5;lane++) {
        field89 t=f89_slot_tweak(kl,kh,q,lane,base);
        u32 *slot=(u32*)encoded+3*lane;
        if (decode) f89_plain_bytes(raw,available,offset+11*lane,f89_read_root(slot,t));
        else f89_write_root(slot,f89_read_plain(raw,available,offset+11*lane),t);
    }
    if (!decode) ((u32*)encoded)[15]=0;
}
#endif
