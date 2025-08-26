# Harris et al. 2003 Nature: Optimal Timescale Analysis using GLM
# Find optimal timescale using generalized linear models and fine temporal resolution
# Implements the exact method from Harris et al. 2003 with Gaussian smearing and penalized log-likelihood
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.optimize import minimize
from scipy.special import loggamma
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

def gaussian_smear_spike_train(spike_times, time_points, sigma):
    """
    Apply Gaussian smearing to spike train as per Harris et al. 2003 Equation (9)
    OPTIMIZED VERSION for speed
    
    Parameters:
    -----------
    spike_times : array
        Spike times in seconds
    time_points : array
        Time points to evaluate smeared activity
    sigma : float
        Gaussian width (peer prediction timescale) in seconds
        
    Returns:
    --------
    array : Smeared activity s_tα at each time point
    """
    if len(spike_times) == 0:
        return np.zeros_like(time_points)
    
    # OPTIMIZATION: Vectorize the calculation for speed
    # Reshape for broadcasting: time_points (n_time,) -> (n_time, 1)
    #                       spike_times (n_spikes,) -> (1, n_spikes)
    time_grid = time_points[:, np.newaxis]  # Shape: (n_time, 1)
    spike_grid = spike_times[np.newaxis, :]  # Shape: (1, n_spikes)
    
    # Vectorized Gaussian kernel calculation
    # Harris et al. 2003 Equation (9): s_tα = (1/√(2πσ²)) * Σ_{τα} exp((t - τα)²/(2σ²))
    kernel_matrix = np.exp(-(time_grid - spike_grid)**2 / (2 * sigma**2))
    
    # Sum across spikes and normalize
    smeared_activity = np.sum(kernel_matrix, axis=1)
    normalization = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    smeared_activity *= normalization
    
    return smeared_activity

def piecewise_link_function(eta):
    """
    Harris et al. 2003 Equation (11): Piecewise link function g(η)
    
    g(η) = { exp(η)  if η < 0
             { η + 1  if η ≥ 0
    
    This prevents excessively high predicted intensities when many positively 
    predicting peer cells are firing simultaneously.
    """
    return np.where(eta < 0, np.exp(eta), eta + 1)

def penalized_log_likelihood(weights, smeared_activities, target_spikes, dt, lambda_reg=0.25):
    """
    Harris et al. 2003 Equation (12): Penalized log-likelihood function L
    
    L = Σ_t [-f_t dt + n_t log(f_t dt)] - (1/4) Σ_α w_α²
    
    Parameters:
    -----------
    weights : array
        Prediction weights w_α for each peer cell
    smeared_activities : array
        Matrix of smeared activities (time x peer_cells)
    target_spikes : array
        Observed spike counts n_t for target neuron
    dt : float
        Time bin size in seconds
    lambda_reg : float
        Regularization parameter (default 1/4 as in Harris et al. 2003)
        
    Returns:
    --------
    float : Negative log-likelihood (for minimization)
    """
    # Calculate predicted intensity f_t = g(Σ_α s_tα w_α)
    linear_predictor = np.dot(smeared_activities, weights)
    predicted_intensity = piecewise_link_function(linear_predictor)
    
    # Log-likelihood term: Σ_t [-f_t dt + n_t log(f_t dt)]
    log_likelihood = 0
    for t in range(len(target_spikes)):
        f_t = predicted_intensity[t]
        n_t = target_spikes[t]
        
        if f_t > 0 and n_t >= 0:
            # Handle log(0) case
            if f_t * dt > 0:
                log_likelihood += -f_t * dt + n_t * np.log(f_t * dt)
            else:
                log_likelihood += -f_t * dt
    
    # Penalty term: -(1/4) Σ_α w_α²
    penalty = -lambda_reg * np.sum(weights**2)
    
    # Return negative for minimization
    return -(log_likelihood + penalty)

def compute_cross_validation_r2(smeared_activities, target_spikes, dt, lambda_reg=0.25, n_folds=5):
    """
    Compute cross-validation R² score using k-fold temporal splitting
    
    Parameters:
    -----------
    smeared_activities : array
        Matrix of smeared peer neuron activities (n_time_points, n_peers)
    target_spikes : array
        Target neuron spike counts for each time bin
    dt : float
        Time bin size in seconds
    lambda_reg : float
        L2 regularization parameter
    n_folds : int
        Number of folds for cross-validation
        
    Returns:
    --------
    float : Cross-validation R² score
    """
    n_time_points = len(target_spikes)
    
    # Ensure we have enough data for cross-validation
    if n_time_points < n_folds * 10:  # Need at least 10 points per fold
        print(f"      ⚠️  WARNING: Not enough data for {n_folds}-fold CV (only {n_time_points} points)")
        return np.nan
    
    # Create temporal folds (preserve temporal order)
    fold_size = n_time_points // n_folds
    cv_scores = []
    
    for fold in range(n_folds):
        # Define validation indices for this fold
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_time_points
        
        # Training indices (all except validation)
        train_indices = list(range(0, val_start)) + list(range(val_end, n_time_points))
        
        if len(train_indices) < 10:  # Need minimum training data
            continue
            
        # Split data
        X_train = smeared_activities[train_indices]
        y_train = target_spikes[train_indices]
        X_val = smeared_activities[val_start:val_end]
        y_val = target_spikes[val_start:val_end]
        
        # Skip if validation set has no spikes
        if np.sum(y_val) == 0:
            continue
            
        try:
            # Train model on training data
            n_peers = X_train.shape[1]
            initial_weights = np.zeros(n_peers)
            
            # Optimize weights using training data
            result = minimize(
                penalized_log_likelihood,
                initial_weights,
                args=(X_train, y_train, dt, lambda_reg),
                method='L-BFGS-B',
                options={'maxiter': 100, 'disp': False}
            )
            
            if result.success:
                # Predict on validation data
                val_weights = result.x
                predicted_intensity = np.exp(X_val @ val_weights)
                
                # Compute R² on validation set
                ss_res = np.sum((y_val - predicted_intensity)**2)
                ss_tot = np.sum((y_val - np.mean(y_val))**2)
                
                if ss_tot > 0:
                    r2_val = 1 - (ss_res / ss_tot)
                    cv_scores.append(r2_val)
                    
        except Exception as e:
            print(f"      ⚠️  WARNING: CV fold {fold} failed: {e}")
            continue
    
    # Return mean CV R² score
    if len(cv_scores) > 0:
        cv_r2 = np.mean(cv_scores)
        print(f"      🔍 DEBUG: CV completed with {len(cv_scores)}/{n_folds} folds, mean R²: {cv_r2:.6f}")
        return cv_r2
    else:
        print(f"      ⚠️  WARNING: No successful CV folds")
        return np.nan

def predict_single_neuron_harris_glm(spike_times, indices_final, target_neuron_idx, sigma, dt, quick_test=False, fast_mode=False, cv_enabled=True, max_time_points=100000):
    """
    Predict single neuron activity using Harris et al. 2003 GLM method
    OPTIMIZED VERSION for speed with comprehensive debugging
    
    Parameters:
    -----------
    spike_times : TsGroup
        Spike times for all neurons
    indices_final : array
        Indices of good neurons
    target_neuron_idx : int
        Index of target neuron to predict
    sigma : float
        Gaussian width (peer prediction timescale) in seconds
    dt : float
        Time bin size in seconds
    quick_test : bool
        If True, use minimal settings for very fast testing
    fast_mode : bool
        If True, use fast mode settings
    cv_enabled : bool
        If True, compute cross-validation R² scores
    max_time_points : int
        Maximum number of time points to allow before subsampling
        
    Returns:
    --------
    tuple : (predicted_intensity, actual_spikes, model_performance)
    """
    print(f"      🔍 DEBUG: Starting prediction for neuron {target_neuron_idx}")
    
    # Get target neuron spike times
    target_unit = spike_times[indices_final[target_neuron_idx]]
    target_spike_times = target_unit.as_series().index.values
    print(f"      🔍 DEBUG: Target neuron has {len(target_spike_times)} spikes")
    
    # Get peer neuron spike times (exclude target)
    peer_indices = [idx for i, idx in enumerate(indices_final) if i != target_neuron_idx]
    peer_spike_times = []
    for idx in peer_indices:
        unit = spike_times[idx]
        spikes = unit.as_series().index.values
        peer_spike_times.append(spikes)
        print(f"      🔍 DEBUG: Peer neuron {idx} has {len(spikes)} spikes")
    
    if len(peer_spike_times) == 0:
        print(f"      ❌ ERROR: No peer neurons found")
        return None, None, {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': 'No peer neurons'}
    
    # Determine time range
    all_times = [target_spike_times] + peer_spike_times
    all_times_flat = np.concatenate(all_times)
    if len(all_times_flat) == 0:
        print(f"      ❌ ERROR: No spikes found in any neuron")
        return None, None, {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': 'No spikes found'}
    
    start_time = np.min(all_times_flat)
    end_time = np.max(all_times_flat)
    print(f"      🔍 DEBUG: Time range: {start_time:.3f}s to {end_time:.3f}s (duration: {end_time-start_time:.3f}s)")
    
    # OPTIMIZATION: Limit time range for very long recordings
    # For quick_test and fast_mode, use full time series but downsample aggressively
    if quick_test or fast_mode:
        mode_name = "QUICK TEST" if quick_test else "FAST MODE"
        print(f"      🔍 DEBUG: {mode_name} - Using FULL time series ({end_time-start_time:.1f}s)")
        # Don't limit time range, let downsampling handle the speed
        max_duration = end_time - start_time  # Use full duration
    else:
        max_duration = 300  # Maximum 5 minutes to analyze for other modes
        
    if end_time - start_time > max_duration:
        if not (quick_test or fast_mode):  # Only limit time for non-optimized modes
            print(f"      ⚠️  Long recording detected ({end_time-start_time:.1f}s), limiting to {max_duration}s")
            # Use the middle portion instead of the end to avoid missing spikes
            mid_time = (start_time + end_time) / 2
            start_time = mid_time - max_duration / 2
            end_time = mid_time + max_duration / 2
            print(f"      🔍 DEBUG: New time range: {start_time:.3f}s to {end_time:.3f}s")
        else:
            mode_name = "QUICK TEST" if quick_test else "FAST MODE"
            print(f"      🔍 DEBUG: {mode_name} - Keeping full time range for maximum spike coverage")
    
    # Create time bins
    time_bins = np.arange(start_time, end_time + dt, dt)
    time_points = time_bins[:-1] + dt/2  # Center of each bin
    print(f"      🔍 DEBUG: Created {len(time_points)} time points with dt={dt*1000:.1f}ms")
    print(f"      🔍 DEBUG: Time bins: {len(time_bins)} bins from {time_bins[0]:.3f}s to {time_bins[-1]:.3f}s")
    
    # OPTIMIZATION: Skip if too many time points (memory/speed issue)
    # Use the max_time_points from the calling function
    if len(time_points) > max_time_points:
        # Subsample time points but ensure we don't lose spikes
        step = len(time_points) // max_time_points
        print(f"      ⚠️  Too many time points ({len(time_points)}), subsampling by factor {step}")
        
        # Store original values for debugging
        original_time_points = time_points.copy()
        original_time_bins = time_bins.copy()
        
        time_points = time_points[::step]
        time_bins = time_bins[::step]  # Also subsample bins to match
        
        print(f"      🔍 DEBUG: After subsampling: {len(time_points)} time points, {len(time_bins)} bins")
        print(f"      🔍 DEBUG: Original dt: {dt*1000:.1f}ms, Effective dt: {dt*step*1000:.1f}ms")
        print(f"      🔍 DEBUG: Original time range: {original_time_bins[0]:.3f}s to {original_time_bins[-1]:.3f}s")
        print(f"      🔍 DEBUG: Subsampled time range: {time_bins[0]:.3f}s to {time_bins[-1]:.3f}s")
        
        # Check if we're losing time coverage
        if time_bins[-1] < end_time:
            print(f"      ⚠️  WARNING: Subsampling reduced time coverage! End time went from {end_time:.3f}s to {time_bins[-1]:.3f}s")
        
        # Show sample of subsampled bins (only if there are issues)
        if time_bins[-1] < end_time:
            print(f"      ⚠️  WARNING: Subsampling reduced time coverage! End time went from {end_time:.3f}s to {time_bins[-1]:.3f}s")
    else:
        print(f"      🔍 DEBUG: No subsampling needed, using all {len(time_points)} time points")
    
    # Create target spike count array
    target_spikes = np.zeros(len(time_points))
    spikes_in_window = 0
    spikes_binned = 0
    
    # Bin spikes into time bins (silent operation)
    for spike_time in target_spike_times:
        if start_time <= spike_time <= end_time:
            spikes_in_window += 1
            # Find which bin this spike belongs to
            bin_idx = np.searchsorted(time_bins, spike_time) - 1
            
            if 0 <= bin_idx < len(target_spikes):
                target_spikes[bin_idx] += 1
                spikes_binned += 1
            else:
                print(f"      ⚠️  WARNING: Spike at {spike_time:.3f}s -> bin {bin_idx} is out of range [0, {len(target_spikes)})")
    
    # Only show summary if there are issues
    if spikes_binned == 0:
        print(f"      ❌ ERROR: No spikes were binned successfully!")
        print(f"      🔍 DEBUG: Spikes in window: {spikes_in_window}, Successfully binned: {spikes_binned}")
    elif spikes_binned < spikes_in_window:
        print(f"      ⚠️  WARNING: Only {spikes_binned}/{spikes_in_window} spikes were binned successfully")
    
    # Check if target has enough spikes
    if np.sum(target_spikes) < 5:
        print(f"      ❌ ERROR: Target neuron has too few spikes ({np.sum(target_spikes)}) for reliable prediction")
        print(f"      🔍 DEBUG: This suggests a problem with the binning or time window selection")
        print(f"      🔍 DEBUG: Possible causes:")
        print(f"      🔍 DEBUG:   1. Time window doesn't contain spikes")
        print(f"      🔍 DEBUG:   2. Subsampling misaligned bins and spikes")
        print(f"      🔍 DEBUG:   3. Bin boundaries don't match spike times")
        return None, None, {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': 'Target neuron has too few spikes'}
    
    # Create smeared activity matrix for peer neurons
    smeared_activities = np.zeros((len(time_points), len(peer_spike_times)))
    for i, peer_spikes in enumerate(peer_spike_times):
        if len(peer_spikes) > 0:  # Only process neurons with spikes
            # Filter spikes to time window before smearing
            window_spikes = peer_spikes[(peer_spikes >= start_time) & (peer_spikes <= end_time)]
            if len(window_spikes) > 0:
                smeared_activities[:, i] = gaussian_smear_spike_train(window_spikes, time_points, sigma)
            else:
                print(f"      ⚠️  WARNING: Peer neuron {i} has no spikes in time window")
        else:
            print(f"      ⚠️  WARNING: Peer neuron {i} has no spikes")
    
    # Check if smeared activities are valid
    if np.all(smeared_activities == 0):
        print(f"      ❌ ERROR: All peer neurons have zero smeared activity")
        return None, None, {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': 'All peer neurons have zero activity'}
    
    # Standardize smeared activities
    smeared_activities_std = stats.zscore(smeared_activities, axis=0)
    
    # Initialize weights
    n_peer_neurons = len(peer_spike_times)
    initial_weights = np.zeros(n_peer_neurons)
    
    try:
        # OPTIMIZATION: Use faster optimization method and limit iterations
        result = minimize(
            penalized_log_likelihood,
            initial_weights,
            args=(smeared_activities_std, target_spikes, dt),
            method='L-BFGS-B',
            options={'maxiter': 100, 'disp': False}  # Reduced from 1000
        )
        
        if not result.success:
            print(f"      ❌ ERROR: Optimization failed: {result.message}")
            return None, None, {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': f'Optimization failed: {result.message}'}
        
        optimal_weights = result.x
        
        # Calculate predicted intensity
        linear_predictor = np.dot(smeared_activities_std, optimal_weights)
        predicted_intensity = piecewise_link_function(linear_predictor)
        
        # Calculate performance metrics
        r2 = r2_score(target_spikes, predicted_intensity)
        mse = mean_squared_error(target_spikes, predicted_intensity)
        
        print(f"      🔍 DEBUG: Performance - R²: {r2:.6f}, MSE: {mse:.6f}")
        
        # Check for valid R²
        if np.isnan(r2) or np.isinf(r2):
            print(f"      ❌ ERROR: Invalid R² value: {r2}")
            return None, None, {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': f'Invalid R² value: {r2}'}
        
        # Cross-validation R²
        if cv_enabled:
            print(f"      🔍 DEBUG: Computing cross-validation performance...")
            cv_r2 = compute_cross_validation_r2(smeared_activities, target_spikes, dt, lambda_reg=0.25, n_folds=5)
            print(f"      🔍 DEBUG: CV R²: {cv_r2:.6f}")
        else:
            print(f"      🔍 DEBUG: Cross-validation disabled, skipping CV computation")
            cv_r2 = np.nan
        
        performance = {
            'r2': r2,
            'mse': mse,
            'cv_r2': cv_r2,
            'sigma': sigma,
            'dt': dt,
            'weights': optimal_weights,
            'method': 'harris_2003_glm'
        }
        
        print(f"      ✅ SUCCESS: Prediction completed successfully")
        return predicted_intensity, target_spikes, performance
        
    except Exception as e:
        print(f"      ❌ ERROR: GLM failed with exception: {str(e)}")
        import traceback
        print(f"      🔍 DEBUG: Full traceback:")
        traceback.print_exc()
        return None, None, {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': f'GLM failed: {str(e)}'}

def analyze_prediction_quality_harris_glm(spike_times, indices_final, sigma, dt, quick_test=False, fast_mode=False, cv_enabled=True, max_time_points=100000):
    """
    Analyze prediction quality across all neurons for a given sigma and dt
    OPTIMIZED VERSION with progress tracking and error handling
    
    Parameters:
    -----------
    spike_times : TsGroup
        Spike times for all neurons
    sigma : float
        Gaussian width in seconds
    dt : float
        Time bin size in seconds
    quick_test : bool
        Whether this is a quick test run
    fast_mode : bool
        Whether this is a fast mode run
    cv_enabled : bool
        Whether to compute cross-validation R² scores
    max_time_points : int
        Maximum number of time points to allow before subsampling
        
    Returns:
    --------
    dict : Performance metrics for each neuron
    """
    print(f"    Analyzing {len(indices_final)} neurons...")
    
    results = {}
    successful_predictions = 0
    
    for i in range(len(indices_final)):
        try:
            predicted_intensity, actual_spikes, performance = predict_single_neuron_harris_glm(
                spike_times, indices_final, i, sigma, dt, quick_test, fast_mode, cv_enabled, max_time_points
            )
            
            if performance and not np.isnan(performance.get('r2', np.nan)):
                results[i] = performance
                successful_predictions += 1
                print(f"      ✅ Neuron {i}: R² = {performance['r2']:.4f}")
            else:
                print(f"      ❌ Neuron {i}: Failed prediction")
                results[i] = {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': 'Prediction failed'}
                
        except Exception as e:
            print(f"      ❌ Neuron {i}: Exception - {str(e)}")
            results[i] = {'r2': np.nan, 'mse': np.nan, 'cv_r2': np.nan, 'error': f'Exception: {str(e)}'}
    
    print(f"    📊 Results: {successful_predictions}/{len(indices_final)} successful predictions")
    
    # Calculate summary statistics
    if successful_predictions > 0:
        r2_values = [results[i]['r2'] for i in results if not np.isnan(results[i]['r2'])]
        mse_values = [results[i]['mse'] for i in results if not np.isnan(results[i]['mse'])]
        cv_r2_values = [results[i]['cv_r2'] for i in results if not np.isnan(results[i]['cv_r2'])]
        
        avg_r2 = np.mean(r2_values)
        avg_mse = np.mean(mse_values)
        avg_cv_r2 = np.mean(cv_r2_values) if cv_r2_values else np.nan
        
        return {
            'sigma': sigma,
            'dt': dt,
            'n_neurons': len(indices_final),
            'avg_r2': avg_r2,
            'avg_mse': avg_mse,
            'avg_cv_r2': avg_cv_r2,
            'individual_results': results,
            'successful_predictions': successful_predictions,
            'failed_predictions': len(indices_final) - successful_predictions,
            'method': 'harris_2003_glm'
        }
    else:
        return {
            'sigma': sigma,
            'dt': dt,
            'n_neurons': len(indices_final),
            'avg_r2': np.nan,
            'avg_mse': np.nan,
            'error': 'No successful predictions',
            'individual_results': results,
            'successful_predictions': 0,
            'failed_predictions': len(indices_final),
            'method': 'harris_2003_glm'
        }

def find_optimal_timescale_harris_glm(spike_times, indices_final, sigma_values, dt_values, quick_test=False, fast_mode=False, cv_enabled=True, max_time_points=100000):
    """
    Find optimal timescale parameters using Harris et al. 2003 GLM method
    OPTIMIZED VERSION with progress tracking
    
    Parameters:
    -----------
    spike_times : TsGroup
        Spike times for all neurons
    indices_final : array
        Indices of good neurons
    sigma_values : list
        List of Gaussian widths (peer prediction timescales) to test (in seconds)
    dt_values : list
        List of time bin sizes to test (in seconds)
    quick_test : bool
        Whether this is a quick test run
    fast_mode : bool
        Whether this is a fast mode run
    cv_enabled : bool
        Whether to compute cross-validation R² scores
    max_time_points : int
        Maximum number of time points to allow before subsampling
        
    Returns:
    --------
    dict : Results for each sigma and dt combination
    """
    results = {}
    
    # Calculate total number of combinations for progress tracking
    total_combinations = len(sigma_values) * len(dt_values)
    completed_combinations = 0
    
    print(f"Starting analysis of {total_combinations} parameter combinations...")
    start_time = time.time()
    
    for sigma in sigma_values:
        results[sigma] = {}
        for dt in dt_values:
            completed_combinations += 1
            elapsed_time = time.time() - start_time
            avg_time_per_combo = elapsed_time / completed_combinations
            remaining_combos = total_combinations - completed_combinations
            estimated_remaining_time = remaining_combos * avg_time_per_combo
            
            print(f"\n[{completed_combinations}/{total_combinations}] "
                  f"({completed_combinations/total_combinations*100:.1f}%) "
                  f"Analyzing Harris GLM: σ={sigma*1000:.1f}ms, dt={dt*1000:.1f}ms")
            print(f"  Elapsed: {elapsed_time:.1f}s, "
                  f"ETA: {estimated_remaining_time:.1f}s")
            
            try:
                quality_metrics = analyze_prediction_quality_harris_glm(
                    spike_times, indices_final, sigma, dt, quick_test, fast_mode, cv_enabled, max_time_points
                )
                results[sigma][dt] = quality_metrics
                
                if 'error' not in quality_metrics:
                    print(f"  ✓ Average R²: {quality_metrics['avg_r2']:.3f}")
                    print(f"  ✓ Average MSE: {quality_metrics['avg_mse']:.3f}")
                    if cv_enabled and 'avg_cv_r2' in quality_metrics and not np.isnan(quality_metrics['avg_cv_r2']):
                        print(f"  ✓ Average CV R²: {quality_metrics['avg_cv_r2']:.3f}")
                    elif cv_enabled:
                        print(f"  ⚠️  No CV R² available")
                    else:
                        print(f"  ⚠️  Cross-validation disabled")
                else:
                    print(f"  ✗ Error: {quality_metrics['error']}")
                    
            except Exception as e:
                print(f"  ✗ Error analyzing σ={sigma*1000:.1f}ms, dt={dt*1000:.1f}ms: {e}")
                continue
    
    total_time = time.time() - start_time
    print(f"\n🎉 Analysis completed in {total_time:.1f}s!")
    print(f"   Total combinations: {total_combinations}")
    print(f"   Successful: {sum(1 for sigma in results for dt in results[sigma] if 'error' not in results[sigma][dt])}")
    print(f"   Failed: {sum(1 for sigma in results for dt in results[sigma] if 'error' in results[sigma][dt])}")
    
    return results

def run_harris_glm_timescale_analysis(spike_times, indices_final, fast_mode=True, quick_test=False, cv_enabled=True):
    """
    Main entry point for Harris et al. 2003 GLM-based optimal timescale analysis
    
    Parameters:
    -----------
    spike_times : TsGroup
        Spike times for all neurons
    indices_final : array
        Indices of good neurons
    fast_mode : bool
        If True, use optimized settings for faster execution
    quick_test : bool
        If True, use minimal settings for very fast testing
    cv_enabled : bool
        If True, compute cross-validation R² scores (slower but more robust)
        
    Returns:
    --------
    dict : Results dictionary with performance metrics for each parameter combination
    """
    if quick_test:
        print("🚀 QUICK TEST MODE: Using minimal settings for very fast execution")
        print("   ⚡ Limited to first 2 neurons for quick testing")
        print("   ⚡ Using FULL time series with aggressive downsampling")
        
        # Use full time series but downsample aggressively
        sigma_values_ms = [10, 25, 50, 100, 200, 500]
        # sigma_values_ms = [s/4 for s in sigma_values_ms]
        dt_values_ms = [5]
        print("   ⚡ Minimal parameter space: σ=2-500ms, dt=1ms")
        
        # Limit neurons for quick testing
        indices_final = indices_final[:2]
        
        # Set aggressive downsampling for quick test
        max_time_points = 400000  # More aggressive than normal 100k
        
    elif fast_mode:
        print("⚡ FAST MODE: Using full recording reduced parameter space")
        sigma_values_ms = [10, 25, 50]
        dt_values_ms = [5]
        max_time_points = 400000 
        
    else:
        print("🐌 FULL MODE: Using comprehensive parameter space")
        sigma_values_ms = [10, 25, 50, 100, 200, 500]
        dt_values_ms = [5]
        max_time_points = 400000
    
    sigma_values = [s/1000.0 for s in sigma_values_ms]
    dt_values = [dt/1000.0 for dt in dt_values_ms]
    
    print("🚀 Starting Harris et al. 2003 GLM-based optimal timescale analysis...")
    print(f"📊 Testing Gaussian widths (σ): {sigma_values_ms} ms")
    print(f"⏱️  Testing time bin sizes (dt): {dt_values_ms} ms")
    print(f"🧠 Number of good neurons: {len(indices_final)}")
    print(f"🔢 Total parameter combinations: {len(sigma_values) * len(dt_values)}")
    print("\n📖 This implements the exact method from Harris et al. 2003:")
    print("   - Gaussian smearing of peer cell spike trains")
    print("   - Piecewise link function g(η) = {exp(η) if η<0, η+1 if η≥0}")
    print("   - Penalized log-likelihood optimization")
    print("\n⚡ OPTIMIZATIONS:")
    print("   - Vectorized Gaussian smearing")
    print("   - Reduced optimization iterations")
    print("   - Time point subsampling if needed")
    if cv_enabled:
        print("   - Cross-validation enabled (5-fold temporal splitting)")
    else:
        print("   - Cross-validation disabled (faster execution)")
    if quick_test:
        print("   - QUICK TEST: Minimal neurons and parameters")
    elif fast_mode:
        print("   - FAST MODE: Full recording with aggressive downsampling")
    print("\n" + "="*60)
    
    results = find_optimal_timescale_harris_glm(spike_times, indices_final, sigma_values, dt_values, quick_test, fast_mode, cv_enabled, max_time_points)
    
    return results

def plot_harris_glm_results(results):
    """
    Plot Harris GLM results showing performance across sigma and dt parameters
    UPDATED: Optimized for single dt value (1ms) scenarios
    
    Parameters:
    -----------
    results : dict
        Results from find_optimal_timescale_harris_glm
    """
    if not results:
        print("❌ No results to plot")
        return
    
    # Extract unique sigma and dt values from nested structure
    sigma_values = []
    dt_values = []
    
    # Handle nested structure: results[sigma][dt] = metrics
    for sigma in results:
        if isinstance(results[sigma], dict):
            sigma_values.append(sigma)
            for dt in results[sigma]:
                if isinstance(results[sigma][dt], dict) and 'sigma' in results[sigma][dt] and 'dt' in results[sigma][dt]:
                    dt_values.append(results[sigma][dt]['dt'])
    
    # Remove duplicates and sort
    sigma_values = sorted(list(set(sigma_values)))
    dt_values = sorted(list(set(dt_values)))
    
    if not sigma_values:
        print("❌ No valid results found for plotting")
        return
    
    print(f"📊 Plotting results for {len(sigma_values)} sigma values and {len(dt_values)} dt values")
    print(f"   Sigma range: {min(sigma_values)*1000:.0f}ms to {max(sigma_values)*1000:.0f}ms")
    print(f"   DT values: {[dt*1000 for dt in dt_values]}ms")
    
    # Debug: Show data types and exact values
    print(f"🔍 DEBUG: Data type analysis:")
    print(f"   Sigma values (raw): {sigma_values}")
    print(f"   Sigma values (ms): {[s*1000 for s in sigma_values]}")
    print(f"   DT values (raw): {dt_values}")
    print(f"   DT values (ms): {[dt*1000 for dt in dt_values]}")
    print(f"   Results keys (raw): {list(results.keys())}")
    print(f"   Results keys (ms): {[k*1000 for k in results.keys()]}")
    
    # Create figure with subplots
    if len(dt_values) == 1:
        # Single dt value - use 1D plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Harris et al. 2003 GLM Results (dt = 1ms)', fontsize=14)
        
        # Extract R², MSE, and CV R² values for plotting
        r2_values = []
        mse_values = []
        cv_r2_values = []
        
        print(f"🔍 DEBUG: Starting data extraction...")
        print(f"   Sigma values to process: {[s*1000 for s in sigma_values]}ms")
        print(f"   DT value to use: {dt_values[0]*1000:.0f}ms")
        
        for i, sigma in enumerate(sigma_values):
            print(f"   🔍 Processing σ={sigma*1000:.0f}ms (index {i})...")
            
            # Find results for this sigma in nested structure
            if sigma in results and isinstance(results[sigma], dict):
                print(f"      ✅ Found sigma {sigma} in results")
                # For single dt, we expect results[sigma][dt] structure
                dt = dt_values[0]  # Single dt value
                if dt in results[sigma] and isinstance(results[sigma][dt], dict):
                    print(f"      ✅ Found dt {dt} in results[{sigma}]")
                    sigma_results = results[sigma][dt]
                    print(f"      🔍 Available keys: {list(sigma_results.keys())}")
                    
                    if 'avg_r2' in sigma_results and not np.isnan(sigma_results['avg_r2']):
                        r2_val = sigma_results['avg_r2']
                        mse_val = sigma_results['avg_mse']
                        cv_r2_val = sigma_results.get('avg_cv_r2', np.nan)
                        r2_values.append(r2_val)
                        mse_values.append(mse_val)
                        cv_r2_values.append(cv_r2_val)
                        print(f"      ✅ Added R²={r2_val:.3f}, MSE={mse_val:.3f}, CV R²={cv_r2_val:.3f}")
                    else:
                        print(f"      ❌ Missing or NaN avg_r2: {sigma_results.get('avg_r2', 'MISSING')}")
                        r2_values.append(np.nan)
                        mse_values.append(np.nan)
                        cv_r2_values.append(np.nan)
                else:
                    print(f"      ❌ dt {dt} not found in results[{sigma}]")
                    print(f"      🔍 Available dt keys: {list(results[sigma].keys())}")
                    r2_values.append(np.nan)
                    mse_values.append(np.nan)
                    cv_r2_values.append(np.nan)
            else:
                print(f"      ❌ Sigma {sigma} not found in results or not a dict")
                print(f"      🔍 Results keys: {list(results.keys())}")
                r2_values.append(np.nan)
                mse_values.append(np.nan)
                cv_r2_values.append(np.nan)
        
        print(f"🔍 DEBUG: Data extraction complete:")
        print(f"   Final r2_values: {r2_values}")
        print(f"   Final mse_values: {mse_values}")
        print(f"   Final cv_r2_values: {cv_r2_values}")
        print(f"   Valid R² count: {sum(1 for r in r2_values if not np.isnan(r))}")
        print(f"   Valid MSE count: {sum(1 for m in mse_values if not np.isnan(m))}")
        print(f"   Valid CV R² count: {sum(1 for r in cv_r2_values if not np.isnan(r))}")
        
        # Check if we have any valid data to plot
        valid_r2_count = sum(1 for r in r2_values if not np.isnan(r))
        if valid_r2_count == 0:
            print("❌ ERROR: No valid R² values found! Cannot create plots.")
            print("🔍 DEBUG: Let me investigate the data structure...")
            
            # Debug: Show what we actually have
            print(f"   Results keys: {list(results.keys())}")
            for sigma in results:
                if isinstance(results[sigma], dict):
                    print(f"   Results[{sigma}] keys: {list(results[sigma].keys())}")
                    for dt in results[sigma]:
                        if isinstance(results[sigma][dt], dict):
                            print(f"   Results[{sigma}][{dt}] keys: {list(results[sigma][dt].keys())}")
                            if 'avg_r2' in results[sigma][dt]:
                                print(f"   Results[{sigma}][{dt}]['avg_r2'] = {results[sigma][dt]['avg_r2']}")
                            else:
                                print(f"   Results[{sigma}][{dt}] missing 'avg_r2' key")
            
            print(f"   Extracted sigma_values: {sigma_values}")
            print(f"   Extracted dt_values: {dt_values}")
            print(f"   Final r2_values: {r2_values}")
            print(f"   Final mse_values: {mse_values}")
            return
        
        print(f"✅ Found {valid_r2_count} valid R² values for plotting")
        
        # Debug: Show exactly what we're plotting
        print(f"🔍 DEBUG: Plotting details:")
        print(f"   X-axis (sigma values in ms): {[s*1000 for s in sigma_values]}")
        print(f"   Y-axis (R² values): {r2_values}")
        print(f"   Y-axis (MSE values): {mse_values}")
        
        # Plot R² vs sigma
        x_vals = [s*1000 for s in sigma_values]
        print(f"   🔍 Plotting R²: x={x_vals}, y={r2_values}")
        print(f"   🔍 Plotting CV R²: x={x_vals}, y={cv_r2_values}")
        
        # Plot training R² values
        ax1.plot(x_vals, r2_values, 'bo-', linewidth=2, markersize=8, label='Training R²')
        
        # Plot cross-validation R² values if available
        valid_cv_mask = ~np.isnan(cv_r2_values)
        if np.any(valid_cv_mask):
            ax1.plot(np.array(x_vals)[valid_cv_mask], np.array(cv_r2_values)[valid_cv_mask], 
                    'rs-', linewidth=2, markersize=8, label='Cross-validation R²')
        
        ax1.set_xlabel('Gaussian Width σ (ms)')
        ax1.set_ylabel('Average R² Score')
        ax1.set_title('Prediction Quality vs Gaussian Width')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        ax1.legend()
        
        # Plot MSE vs sigma
        print(f"   🔍 Plotting MSE: x={x_vals}, y={mse_values}")
        ax2.plot(x_vals, mse_values, 'ro-', linewidth=2, markersize=8, label='MSE values')
        ax2.set_xlabel('Gaussian Width σ (ms)')
        ax2.set_ylabel('Average MSE')
        ax2.set_title('Prediction Error vs Gaussian Width')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add value labels on points
        print(f"   🔍 Adding value labels to plots...")
        for i, (sigma, r2, mse, cv_r2) in enumerate(zip(sigma_values, r2_values, mse_values, cv_r2_values)):
            if not np.isnan(r2):
                print(f"      Adding R² label: {r2:.3f} at position ({sigma*1000}, {r2})")
                ax1.annotate(f'{r2:.3f}', (sigma*1000, r2), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9)
            if not np.isnan(cv_r2):
                print(f"      Adding CV R² label: {cv_r2:.3f} at position ({sigma*1000}, {cv_r2})")
                ax1.annotate(f'CV:{cv_r2:.3f}', (sigma*1000, cv_r2), textcoords="offset points", 
                            xytext=(0,-15), ha='center', fontsize=8, color='red')
            if not np.isnan(mse):
                print(f"      Adding MSE label: {mse:.3f} at position ({sigma*1000}, {mse})")
                ax2.annotate(f'{mse:.3f}', (sigma*1000, mse), textcoords="offset points", 
                            xytext=(0,10), ha='center', fontsize=9)
        
        print(f"   🔍 Final plot setup complete")
        
    else:
        # Multiple dt values - use 2D heatmaps (original logic)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Harris et al. 2003 GLM Results', fontsize=16)
        
        # Create 2D arrays for heatmaps
        r2_matrix = np.full((len(sigma_values), len(dt_values)), np.nan)
        mse_matrix = np.full((len(sigma_values), len(dt_values)), np.nan)
        
        for i, sigma in enumerate(sigma_values):
            for j, dt in enumerate(dt_values):
                # Find results for this sigma/dt combination in nested structure
                if sigma in results and isinstance(results[sigma], dict):
                    if dt in results[sigma] and isinstance(results[sigma][dt], dict):
                        combo_results = results[sigma][dt]
                        if 'avg_r2' in combo_results and not np.isnan(combo_results['avg_r2']):
                            r2_matrix[i, j] = combo_results['avg_r2']
                            mse_matrix[i, j] = combo_results['avg_mse']
        
        # Plot R² heatmap
        im1 = ax1.imshow(r2_matrix, cmap='viridis', aspect='auto', 
                         extent=[min(dt_values)*1000, max(dt_values)*1000, 
                                min(sigma_values)*1000, max(sigma_values)*1000])
        ax1.set_xlabel('Time Bin Size dt (ms)')
        ax1.set_ylabel('Gaussian Width σ (ms)')
        ax1.set_title('R² Score Heatmap')
        plt.colorbar(im1, ax=ax1, label='R² Score')
        
        # Plot MSE heatmap
        im2 = ax2.imshow(mse_matrix, cmap='plasma', aspect='auto',
                         extent=[min(dt_values)*1000, max(dt_values)*1000, 
                                min(sigma_values)*1000, max(sigma_values)*1000])
        ax2.set_xlabel('Time Bin Size dt (ms)')
        ax2.set_ylabel('Gaussian Width σ (ms)')
        ax2.set_title('MSE Heatmap')
        plt.colorbar(im2, ax=ax2, label='MSE')
        
        # Plot R² vs sigma for different dt values
        for j, dt in enumerate(dt_values):
            r2_values = r2_matrix[:, j]
            valid_mask = ~np.isnan(r2_values)
            if np.any(valid_mask):
                ax3.plot([s*1000 for s in np.array(sigma_values)[valid_mask]], 
                        r2_values[valid_mask], 'o-', label=f'dt={dt*1000:.0f}ms')
        ax3.set_xlabel('Gaussian Width σ (ms)')
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² vs σ for Different dt Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot R² vs dt for different sigma values
        for i, sigma in enumerate(sigma_values):
            r2_values = r2_matrix[i, :]
            valid_mask = ~np.isnan(r2_values)
            if np.any(valid_mask):
                ax4.plot([dt*1000 for dt in np.array(dt_values)[valid_mask]], 
                        r2_values[valid_mask], 'o-', label=f'σ={sigma*1000:.0f}ms')
        ax4.set_xlabel('Time Bin Size dt (ms)')
        ax4.set_ylabel('R² Score')
        ax4.set_title('R² vs dt for Different σ Values')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n📊 RESULTS SUMMARY:")
    if len(dt_values) == 1:
        print(f"   Time bin size: {dt_values[0]*1000:.0f}ms (fixed)")
        print(f"   Best σ: {sigma_values[np.nanargmax(r2_values)]*1000:.0f}ms")
        print(f"   Best R²: {np.nanmax(r2_values):.3f}")
        print(f"   Best MSE: {np.nanmin(mse_values):.3f}")
    else:
        best_idx = np.unravel_index(np.nanargmax(r2_matrix), r2_matrix.shape)
        print(f"   Best σ: {sigma_values[best_idx[0]]*1000:.0f}ms")
        print(f"   Best dt: {dt_values[best_idx[1]]*1000:.0f}ms")
        print(f"   Best R²: {r2_matrix[best_idx]:.3f}")
        print(f"   Best MSE: {mse_matrix[best_idx]:.3f}")

def find_optimal_timescale_parameters(results):
    """
    Find optimal timescale parameters (σ and dt) based on Harris GLM results
    """
    print("\n=== Optimal Timescale Parameters (Harris et al. 2003 GLM) ===")
    
    if not results:
        print("No results available")
        return
    
    # Find best combination of σ and dt
    best_r2 = -np.inf
    best_sigma = None
    best_dt = None
    
    for sigma in results:
        for dt in results[sigma]:
            result = results[sigma][dt]
            if 'error' not in result and not np.isnan(result['avg_r2']):
                if result['avg_r2'] > best_r2:
                    best_r2 = result['avg_r2']
                    best_sigma = sigma
                    best_dt = dt
    
    if best_sigma is not None:
        print(f"\nOptimal parameters:")
        print(f"  Gaussian width (σ): {best_sigma*1000:.1f} ms")
        print(f"  Time bin size (dt): {best_dt*1000:.1f} ms")
        print(f"  Best R²: {best_r2:.3f}")
        
        # Show results for optimal parameters
        optimal_result = results[best_sigma][best_dt]
        print(f"  Average MSE: {optimal_result['avg_mse']:.3f}")
        print(f"  Number of neurons: {optimal_result['n_neurons']}")
        
        # Show individual neuron performance
        if 'individual_r2' in optimal_result:
            r2_values = optimal_result['individual_r2']
            print(f"  Individual neuron R² range: {np.min(r2_values):.3f} to {np.max(r2_values):.3f}")
            print(f"  Standard deviation of R²: {np.std(r2_values):.3f}")
    else:
        print("No valid results found")

def diagnose_data_quality(spike_times, indices_final):
    """
    Diagnose data quality issues that might cause prediction failures
    
    Parameters:
    -----------
    spike_times : TsGroup
        Spike times for all neurons
    indices_final : array
        Indices of good neurons
        
    Returns:
    --------
    dict : Diagnostic information
    """
    print("🔍 DIAGNOSING DATA QUALITY...")
    print("=" * 50)
    
    n_neurons = len(indices_final)
    print(f"📊 Total neurons to analyze: {n_neurons}")
    
    if n_neurons < 2:
        print("❌ ERROR: Need at least 2 neurons for prediction")
        return {'status': 'error', 'message': 'Insufficient neurons'}
    
    # Check each neuron
    neuron_stats = []
    total_spikes = 0
    
    for i, idx in enumerate(indices_final):
        try:
            unit = spike_times[idx]
            spikes = unit.as_series().index.values
            n_spikes = len(spikes)
            total_spikes += n_spikes
            
            if n_spikes == 0:
                status = "❌ NO SPIKES"
            elif n_spikes < 5:
                status = "⚠️  VERY FEW SPIKES"
            elif n_spikes < 20:
                status = "⚠️  FEW SPIKES"
            else:
                status = "✅ GOOD"
            
            neuron_stats.append({
                'neuron_idx': i,
                'unit_idx': idx,
                'n_spikes': n_spikes,
                'status': status
            })
            
            print(f"  Neuron {i+1}/{n_neurons} (Unit {idx}): {n_spikes} spikes - {status}")
            
        except Exception as e:
            print(f"  ❌ ERROR accessing neuron {i} (Unit {idx}): {e}")
            neuron_stats.append({
                'neuron_idx': i,
                'unit_idx': idx,
                'n_spikes': 0,
                'status': f"❌ ERROR: {e}"
            })
    
    print("\n📈 SUMMARY STATISTICS:")
    print(f"  Total spikes across all neurons: {total_spikes}")
    print(f"  Average spikes per neuron: {total_spikes/n_neurons:.1f}")
    
    # Check for neurons with sufficient spikes
    neurons_with_spikes = [n for n in neuron_stats if n['n_spikes'] >= 5]
    neurons_without_spikes = [n for n in neuron_stats if n['n_spikes'] == 0]
    
    print(f"  Neurons with ≥5 spikes: {len(neurons_with_spikes)}/{n_neurons}")
    print(f"  Neurons with 0 spikes: {len(neurons_without_spikes)}/{n_neurons}")
    
    # Check time range
    all_spike_times = []
    for n in neuron_stats:
        if n['n_spikes'] > 0:
            try:
                unit = spike_times[indices_final[n['neuron_idx']]]
                spikes = unit.as_series().index.values
                all_spike_times.extend(spikes)
            except:
                continue
    
    if all_spike_times:
        min_time = min(all_spike_times)
        max_time = max(all_spike_times)
        duration = max_time - min_time
        print(f"  Recording duration: {duration:.3f}s ({duration/60:.1f} minutes)")
        
        if duration < 1.0:
            print("  ⚠️  WARNING: Very short recording (<1s)")
        elif duration < 10.0:
            print("  ⚠️  WARNING: Short recording (<10s)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if len(neurons_with_spikes) < 2:
        print("  ❌ Need at least 2 neurons with ≥5 spikes for prediction")
        return {'status': 'error', 'message': 'Insufficient active neurons'}
    
    elif len(neurons_without_spikes) > 0:
        print(f"  ⚠️  Consider excluding {len(neurons_without_spikes)} neurons with 0 spikes")
        print("  💡 Suggestion: Filter indices_final to only include neurons with spikes")
    
    if total_spikes < 50:
        print("  ⚠️  Very low spike count - may affect prediction quality")
    
    print("  ✅ Data appears suitable for analysis")
    
    return {
        'status': 'success',
        'n_neurons': n_neurons,
        'total_spikes': total_spikes,
        'neurons_with_spikes': len(neurons_with_spikes),
        'neurons_without_spikes': len(neurons_without_spikes),
        'neuron_stats': neuron_stats
    }

# Example usage:
if __name__ == "__main__":
    print("Harris et al. 2003 GLM-Based Timescale Analysis")
    print("This script implements the exact method from Harris et al. 2003:")
    print("- Gaussian smearing of peer cell spike trains")
    print("- Piecewise link function g(η)")
    print("- Penalized log-likelihood optimization")
    print("\nUse this script by importing the functions into your notebook:")
    print("from harris_2003_glm_timescale import run_harris_glm_timescale_analysis")
    print("results = run_harris_glm_timescale_analysis(spike_times, indicesFinal)")
    print("\nOptional parameters:")
    print("- fast_mode=True: Use optimized settings for faster execution")
    print("- quick_test=False: Use minimal settings for very fast testing")
    print("- cv_enabled=True: Compute cross-validation R² scores (slower but more robust)")

