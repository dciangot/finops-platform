#ifndef C_METRICS_H
#define C_METRICS_H

/*
 * FinOps Backtest Engine - C Metrics Library
 *
 * Provides core financial metrics calculations for cloud cost optimization
 * analysis. All functions operate on plain C types for CFFI compatibility.
 */

/**
 * Calculate the percentage of savings achieved by an optimization.
 *
 * @param original_cost  The baseline cost before optimization.
 * @param optimized_cost The cost after applying the optimization.
 * @return Savings as a percentage of original_cost (0-100+). Returns 0 if
 *         original_cost <= 0.
 */
double savings_rate(double original_cost, double optimized_cost);

/**
 * Calculate cost efficiency as the ratio of baseline to actual cost.
 * Values > 1 mean the optimized path is cheaper than baseline.
 *
 * @param actual_cost    The actual (current/optimized) cost.
 * @param baseline_cost  The reference baseline cost.
 * @return Efficiency ratio. Returns 0 if actual_cost <= 0.
 */
double cost_efficiency(double actual_cost, double baseline_cost);

/**
 * Calculate Return on Investment (ROI) for a cost optimization initiative.
 *
 * @param total_savings     Total monetary savings achieved.
 * @param total_investment  Total cost of implementing the optimization.
 * @return ROI as a percentage. Returns 0 if total_investment <= 0.
 */
double roi(double total_savings, double total_investment);

/**
 * Calculate the payback period (in months) for an investment.
 *
 * @param investment       Upfront investment cost.
 * @param monthly_savings  Monthly savings achieved.
 * @return Number of months to break even. Returns -1 if monthly_savings <= 0.
 */
double payback_period(double investment, double monthly_savings);

/**
 * Compute the arithmetic mean of an array of cost values.
 *
 * @param values  Pointer to array of double values.
 * @param n       Number of elements in the array.
 * @return Mean value. Returns 0 if n <= 0.
 */
double array_mean(const double *values, int n);

/**
 * Compute the population variance of an array of cost values.
 *
 * @param values  Pointer to array of double values.
 * @param n       Number of elements in the array.
 * @return Population variance. Returns 0 if n <= 0.
 */
double array_variance(const double *values, int n);

/**
 * Compute the population standard deviation of an array of cost values.
 *
 * @param values  Pointer to array of double values.
 * @param n       Number of elements in the array.
 * @return Standard deviation. Returns 0 if n <= 0.
 */
double array_stddev(const double *values, int n);

/**
 * Compute a z-score anomaly score for a single cost observation.
 * A score with |value| > 2 indicates a potential anomaly (2-sigma rule).
 *
 * @param cost    The cost value to evaluate.
 * @param mean    The mean of the reference distribution.
 * @param stddev  The standard deviation of the reference distribution.
 * @return Z-score. Returns 0 if stddev <= 0.
 */
double anomaly_score(double cost, double mean, double stddev);

/**
 * Estimate the linear trend (slope) over a time series of cost values using
 * ordinary least-squares regression with time steps 0, 1, ..., n-1.
 * A positive slope indicates growing costs; negative means declining costs.
 *
 * @param values  Pointer to array of double values ordered by time.
 * @param n       Number of elements in the array.
 * @return Slope of the best-fit line (cost units per time step). Returns 0
 *         if n <= 1.
 */
double trend_slope(const double *values, int n);

/**
 * Compute cumulative savings over a period.
 *
 * @param original   Array of original (pre-optimization) costs per period.
 * @param optimized  Array of optimized costs per period.
 * @param n          Number of periods.
 * @return Total savings summed across all periods.
 */
double cumulative_savings(const double *original, const double *optimized, int n);

#endif /* C_METRICS_H */
