// Shared scan/refine grid contract. The half-space S velocity caps the trapped
// surface-wave domain; matching range and geometry are checked by the bindings.
#ifndef MODISP_COMMON_SCAN_GRID_HPP
#define MODISP_COMMON_SCAN_GRID_HPP

#include "dunkin.hpp"  // MD_HD

namespace md {

// Grid spacing.  Stage 2 maps a grid index back to an exact c, so both sides
// have to agree on where the grid points ARE, not merely how many there are --
// hence one definition rather than two equivalent-looking expressions.
template <typename T>
MD_HD T scan_dc(T cmin, T cmax, int nc) {
  return (nc > 1) ? (cmax - cmin) / static_cast<T>(nc - 1) : T(0);
}

// Highest grid index Stage 2 may consume.
template <typename T>
MD_HD int scan_i_last(T c_cap, T cmin, T cmax, int nc) {
  int i = static_cast<int>((c_cap - cmin) / scan_dc<T>(cmin, cmax, nc));
  if (i > nc - 2) i = nc - 2;
  return i;
}

// Highest grid index Stage 1 must WRITE for Stage 2 to be correct.
//
// One past scan_i_last, not equal to it.  Every read path tops out at
// i_last + 1: the walk classifies the cell (i, i+1) and runs while i <= i_last,
// so it reads row[i_last + 1]; the trough window reads row[tc + 1] with
// tc <= i_last; and retry_bracket reads row[c2 + 1] with c2 <= i_last.
template <typename T>
MD_HD int scan_last_needed(T c_cap, T cmin, T cmax, int nc) {
  return scan_i_last<T>(c_cap, cmin, cmax, nc) + 1;
}

}  // namespace md

#endif  // MODISP_COMMON_SCAN_GRID_HPP
