// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Grid all-to-one barrier. Every non-target node does one remote atomic increment of the target
// node's barrier semaphore over the NoC; the target waits for all `num_signalers` increments (one
// down() per signal) and then records completion in L1 for the host to verify. Exercises max
// semaphore fan-in onto a single counter. Non-DFB: pure cross-node semaphore + NoC.

#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t target_noc_x = get_arg(args::remote_noc_x);
    const uint32_t target_noc_y = get_arg(args::remote_noc_y);
    const uint32_t is_target = get_arg(args::is_target);
    const uint32_t num_signalers = get_arg(args::num_elements);
    const uint32_t result_addr = get_arg(args::result_addr);

    Noc noc;
    Semaphore barrier_sem(sem::barrier_sem);

    if (is_target) {
        // Wait for every signaler's increment (one down() each), then record completion.
        for (uint32_t i = 0; i < num_signalers; ++i) {
            barrier_sem.down(1);
        }
        CoreLocalMem<uint32_t> result(result_addr);
        result[0] = num_signalers;
        flush_l2_cache_line(result_addr);
    } else {
        // Signal the target exactly once.
        barrier_sem.up(noc, target_noc_x, target_noc_y, 1);
    }
}
