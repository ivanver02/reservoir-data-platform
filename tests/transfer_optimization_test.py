"""Tests for transfer_optimization cost matrix incremental update logic.

  1. Build initial state for reservoirs in Cádiz province.
  2. Apply an equal-percentage transfer (donor=375, receiver=212).
  3. Recompute full cost matrix from scratch on updated reservoirs.
  4. Incrementally update the previous cost matrix with update_cost_matrix.
  5. Assert the max absolute finite difference < 1e-15.

If required reservoir IDs or province data are missing, the test is skipped
to avoid false negatives in alternative datasets.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure root on path
sys.path.append(str(Path.cwd()))

from backend.transferring import transfer_optimization as topt


@pytest.mark.parametrize("province,donor,receiver", [("cadiz", 375, 212)])
def test_update_cost_matrix_consistency(province, donor, receiver):
	# Validate province exists
	reservoirs_pd = topt.reservoirs_pd
	if 'province' not in reservoirs_pd.columns:
		pytest.skip("'province' column missing in reservoirs dataset")

	province_mask = reservoirs_pd['province'] == province
	if not province_mask.any():
		pytest.skip(f"Province '{province}' not present in dataset")

	reservoirs_ids = sorted(reservoirs_pd.loc[province_mask, 'id'].unique())
	if donor not in reservoirs_ids or receiver not in reservoirs_ids:
		pytest.skip(f"Required reservoir ids {donor},{receiver} not both in province '{province}' set")

	reservoirs_definitive, cost_matrix, distance_matrix = topt.initial_state(reservoirs_ids)

	updated_reservoirs, _ = topt.update_reservoirs_equal_percentage(reservoirs_definitive, donor, receiver)
	recalculated_cost_matrix, _ = topt.calculate_cost_matrix(updated_reservoirs)

	incrementally_updated_cost_matrix = topt.update_cost_matrix(donor, receiver, cost_matrix.copy(), distance_matrix, updated_reservoirs)

	# Difference ignoring inf / nan
	delta = recalculated_cost_matrix - incrementally_updated_cost_matrix
	finite_mask = np.isfinite(delta)
	if not finite_mask.any():
		pytest.fail("No finite values to compare in cost matrices")
	max_abs_diff = np.abs(delta[finite_mask]).max()

	assert max_abs_diff < 1e-15, f"Max abs diff {max_abs_diff} exceeds tolerance 1e-15"


if __name__ == '__main__': 
	sys.exit(_pytest.main([__file__]))
