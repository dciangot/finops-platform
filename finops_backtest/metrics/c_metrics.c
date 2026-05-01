/*
 * FinOps Backtest Engine - C Metrics Implementation
 *
 * Core financial metrics calculations for cloud cost optimization analysis.
 * Designed for use via Python CFFI bindings.
 */

#include "c_metrics.h"
#include <math.h>

double savings_rate(double original_cost, double optimized_cost) {
    if (original_cost <= 0.0)
        return 0.0;
    return ((original_cost - optimized_cost) / original_cost) * 100.0;
}

double cost_efficiency(double actual_cost, double baseline_cost) {
    if (actual_cost <= 0.0)
        return 0.0;
    return baseline_cost / actual_cost;
}

double roi(double total_savings, double total_investment) {
    if (total_investment <= 0.0)
        return 0.0;
    return ((total_savings - total_investment) / total_investment) * 100.0;
}

double payback_period(double investment, double monthly_savings) {
    if (monthly_savings <= 0.0)
        return -1.0;
    return investment / monthly_savings;
}

double array_mean(const double *values, int n) {
    if (n <= 0)
        return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += values[i];
    return sum / (double)n;
}

double array_variance(const double *values, int n) {
    if (n <= 0)
        return 0.0;
    double mean = array_mean(values, n);
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (double)n;
}

double array_stddev(const double *values, int n) {
    return sqrt(array_variance(values, n));
}

double anomaly_score(double cost, double mean, double stddev) {
    if (stddev <= 0.0)
        return 0.0;
    return (cost - mean) / stddev;
}

double trend_slope(const double *values, int n) {
    if (n <= 1)
        return 0.0;
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < n; i++) {
        double x = (double)i;
        sum_x  += x;
        sum_y  += values[i];
        sum_xy += x * values[i];
        sum_x2 += x * x;
    }
    double denom = (double)n * sum_x2 - sum_x * sum_x;
    if (denom == 0.0)
        return 0.0;
    return ((double)n * sum_xy - sum_x * sum_y) / denom;
}

double cumulative_savings(const double *original, const double *optimized, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++)
        total += original[i] - optimized[i];
    return total;
}
