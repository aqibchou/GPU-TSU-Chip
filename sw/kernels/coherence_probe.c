/* Cross-hart store->load coherence probe (HANDOFF V6.4, hypothesis
   H2): does a word stored by one hart become visible to another
   hart's loads through the SoC memory face, with no fence, and how
   quickly? Decides whether a cross-hart worker pool (t5 lever #1)
   is possible on this silicon at all.

   Layout (arg0 = u32* sc, >= 7 words, zeroed by the host):
     sc[0]  ping    tid0 writes round r as 0xA5000000|r
     sc[1]  pong    warp 1 writes round r as 0x5A000000|r
     sc[2]  verdict tid0: 1 = all rounds acked, 2 = timed out
     sc[3]  rounds acked (tid0)
     sc[4]  total tid0 spin iterations (visibility-latency proxy)
     sc[5]  pings seen by warp 1
     sc[6]  warp-1 timeout flag (2 = gave up waiting for a ping)

   Shape rules (the v6 worker-pool lesson): warp 0 = tids 0-7 share
   one divergence stack, so tid0 runs the ping side while tids 1-7
   return at once (the v5-proven single-active-lane shape). Warp 1 =
   tids 8-15 runs the pong side WARP-UNIFORM: all 8 lanes execute
   the identical loop over the same shared words — every branch
   decision is identical by construction, and the lane-serial memory
   unit serializes their 8 equal-valued stores. Warps 2-7 return at
   once. NEITHER side spins unbounded: both carry a spin budget and
   write a verdict word, so the harness always gets all-64 DONE and
   data — never a bare timeout. Compiles at the -O0 default
   (divergence-legal per runtime_spec §4). */
#include "mk.h"

#define ROUNDS 4u
#define BOUND  50000u

void kernel_main(u32 tid) {
    volatile u32 *sc = (volatile u32 *)mk_arg(0);
    if (tid == 0u) {
        u32 spins = 0u;
        for (u32 r = 1u; r <= ROUNDS; r++) {
            u32 want = 0x5A000000u | r;
            u32 z;
            sc[0] = 0xA5000000u | r;              /* ping */
            for (z = 0u; z < BOUND; z++) {
                if (sc[1] == want) break;
                spins++;
            }
            sc[4] = spins;
            if (z == BOUND) {
                sc[2] = 2u;                       /* pong never seen */
                return;
            }
            sc[3] = r;
        }
        sc[2] = 1u;                               /* all rounds acked */
    } else if ((tid >> 3) == 1u) {
        /* warp 1, all 8 lanes uniform */
        for (u32 r = 1u; r <= ROUNDS; r++) {
            u32 want = 0xA5000000u | r;
            u32 z;
            for (z = 0u; z < BOUND; z++) {
                if (sc[0] == want) break;
            }
            if (z == BOUND) {
                sc[6] = 2u;                       /* ping never seen */
                return;
            }
            sc[5] = r;
            sc[1] = 0x5A000000u | r;              /* pong (8 equal stores) */
        }
    }
}
