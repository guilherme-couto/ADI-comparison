/*-----------------------------------------------------
Monodomain with FitzHugh-Nagumo model
Author: Guilherme Couto
FISIOCOMP - UFJF
------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

/*-----------------------------------------------------
Model parameters
Based on Gerardo_Giorda 2007
-----------------------------------------------------*/
double G = 1.5;         // omega^-1 * cm^-2
double eta1 = 4.4;      // omega^-1 * cm^-1
double eta2 = 0.012;    // dimensionless
double eta3 = 1.0;      // dimensionless
double vth = 13.0;      // mV
double vp = 100.0;      // mV
double ga = 1.2e-3;     // omega^-1 * cm^-1
double chi = 1.0e3;     // cm^-1
double Cm = 1.0e-3;     // mF * cm^-2

/*-----------------------------------------------------
Auxiliary functions
-----------------------------------------------------*/
double reaction_v(double v, double w)
{
    return (1.0 / (Cm * chi)) * ((-G * v * (1.0 - (v / vth)) * (1.0 - (v / vp))) + (-eta1 * v * w));
}

double reaction_w(double v, double w)
{
    return eta2 * ((v / vp) - (eta3 * w));
}

void thomas_algorithm(double *d, double *solution, unsigned long N, double alpha)
{
    // Auxiliary arrays
    double *c_ = (double *)malloc((N - 1) * sizeof(double));
    double *d_ = (double *)malloc((N) * sizeof(double));

    // Coefficients
    double a = -alpha;    // subdiagonal
    double b = 1 + alpha; // diagonal (1st and last row)
    double c = -alpha;    // superdiagonal

    // 1st: update auxiliary arrays
    c_[0] = c / b;
    d_[0] = d[1] / b;

    b = 1 + 2 * alpha;

    for (int i = 1; i <= N - 2; i++)
    {
        c_[i] = c / (b - a * c_[i - 1]);
        d_[i] = (d[i + 1] - a * d_[i - 1]) / (b - a * c_[i - 1]);
    }

    b = 1 + alpha;
    d_[N - 1] = (d[N] - a * d_[N - 2]) / (b - a * c_[N - 2]);

    // 2nd: update solution
    solution[N] = d_[N - 1];

    for (int i = N - 2; i >= 0; i--)
    {
        solution[i + 1] = d_[i] - c_[i] * solution[i + 2];
    }

    // Free memory
    free(c_);
    free(d_);
}

// Adapted for 2nd order approximation
void thomas_algorithm_2nd(double *d, double *solution, unsigned long N, double alpha)
{   
    // Auxiliary arrays
    double *c_ = (double *)malloc((N - 1) * sizeof(double));
    double *d_ = (double *)malloc((N) * sizeof(double));

    // Coefficients
    double a = -alpha;    // subdiagonal
    double b = 1 + 2 * alpha; // diagonal (1st and last row)
    double c = - 2 * alpha;    // superdiagonal
    
    // 1st: update auxiliary arrays
    c_[0] = c / b;
    d_[0] = d[0] / b;

    c = -alpha;
    
    for (int i = 1; i <= N - 2; i++)
    {
        c_[i] = c / (b - a * c_[i - 1]);
        d_[i] = (d[i] - a * d_[i - 1]) / (b - a * c_[i - 1]);
    }
    
    a = - 2 * alpha;
    d_[N - 1] = (d[N - 1] - a * d_[N - 2]) / (b - a * c_[N - 2]);

    a = -alpha;

    // 2nd: update solution
    solution[N - 1] = d_[N - 1];
    
    for (int i = N - 2; i >= 0; i--)
    {
        solution[i] = d_[i] - c_[i] * solution[i + 1];
    }
    
    // Free memory
    free(c_);
    free(d_);
}

// Adapted for 2nd order approximation
double diffusion_i_2nd(int i, int j, int N, double **v)
{
    double result = 0.0;
    if (i == 0)
    {
        result = - 2.0*v[i][j] + 2.0*v[i + 1][j]; 
    }
    else if (i == N - 1)
    {
        result = 2.0*v[i - 1][j] - 2.0*v[i][j]; 
    }
    else
    {
        result = v[i - 1][j] - 2.0*v[i][j] + v[i + 1][j];
    }

    return result;
}

// Adapted for 2nd order approximation
double diffusion_j_2nd(int i, int j, int N, double **v)
{
    double result = 0.0;
    if (j == 0)
    {
        result = - 2.0*v[i][j] + 2.0*v[i][j + 1]; 
    }
    else if (j == N - 1)
    {
        result = 2.0*v[i][j - 1] - 2.0*v[i][j]; 
    }
    else
    {
        result = v[i][j - 1] - 2.0*v[i][j] + v[i][j + 1];
    }

    return result;
}


/*-----------------------------------------------------
Simulation parameters
-----------------------------------------------------*/
int L = 2;              // Length of each side (cm)
double dx = 0.02;       // Spatial step -> cm
double dy = 0.02;       // Spatial step -> cm
double T = 400.0;         // Simulation time -> ms


/*-----------------------------------------------------
Stimulation parameters
-----------------------------------------------------*/
double stim_strength = 100.0;         

double t_s1_begin = 0.0;            // Stimulation start time -> ms
double stim_duration = 1.0;         // Stimulation duration -> ms
double s1_x_limit = 0.2;           // Stimulation x limit -> cm

double t_s2_begin = 120.0;            // Stimulation start time -> ms
double stim2_duration = 3.0;        // Stimulation duration -> ms
double s2_x_max = 1.0;              // Stimulation x max -> cm
double s2_y_max = 1.0;              // Stimulation y limit -> cm
double s2_x_min = 0.0;              // Stimulation x min -> cm
double s2_y_min = 0.0;              // Stimulation y min -> cm


/*-----------------------------------------------------
Main function
-----------------------------------------------------*/
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s <num_threads> <delta_t (ms)> <method>\n", argv[0]);
        exit(1);
    }

    int num_threads = atoi(argv[1]);
    double dt = atof(argv[2]);
    char *method = argv[3];

    if (num_threads <= 0)
    {
        fprintf(stderr, "Number of threads must greater than 0\n");
        exit(1);
    }
    if (strcmp(method, "ADI1") != 0 && strcmp(method, "ADI2") != 0 && strcmp(method, "FE") != 0)
    {
        fprintf(stderr, "Method must be ADI1 (first order) or ADI2 (second order)\n");
        exit(1);
    }

    // Number of steps
    int N = (int)(L / dx);          // Number of spatial steps (square tissue)
    int M = (int)(T / dt);     // Number of time steps

    // Variables
    double **v, **w, **v_tilde, **w_tilde, **r_v, **rightside, **solution;
    v = (double **)malloc(N * sizeof(double *));
    w = (double **)malloc(N * sizeof(double *));
    v_tilde = (double **)malloc(N * sizeof(double *));
    w_tilde = (double **)malloc(N * sizeof(double *));
    r_v = (double **)malloc(N * sizeof(double *));
    rightside = (double **)malloc(N * sizeof(double *));
    solution = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
    {
        v[i] = (double *)malloc(N * sizeof(double));
        w[i] = (double *)malloc(N * sizeof(double));
        v_tilde[i] = (double *)malloc(N * sizeof(double));
        w_tilde[i] = (double *)malloc(N * sizeof(double));
        r_v[i] = (double *)malloc(N * sizeof(double));
        rightside[i] = (double *)malloc(N * sizeof(double));
        solution[i] = (double *)malloc(N * sizeof(double));
    }

    double dv_dt, dw_dt, diff_term = 0.0;

    int step = 0;
    double tstep = 0.0;
    double *time = (double *)malloc(M * sizeof(double));
    int n;        // Time index
    for (n = 0; n < M; n++)
    {
        time[n] = n * dt;
    }

    double I_stim = 0.0;
    int x_lim = s1_x_limit / dx;
    int x_max = s2_x_max / dx;
    int x_min = s2_x_min / dx;
    int y_max = N;
    int y_min = N - s2_y_max / dy;

    double D = ga / (chi * Cm);             // Diffusion coefficient - isotropic
    double phi = D * dt / (dx * dx);        // For Thomas algorithm - isotropic

    // Initial conditions
    int i, j;                       // Spatial indexes i for y-axis and j for x-axis
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            v[i][j] = 0.0;
            w[i][j] = 0.0;
            v_tilde[i][j] = 0.0;
            w_tilde[i][j] = 0.0;
            r_v[i][j] = 0.0;
            rightside[i][j] = 0.0;
            solution[i][j] = 0.0;
        }
    }

    // Prepare files to save data
    // Convert dt to string
    char s_dt[10];
    sprintf(s_dt, "%.03f", dt);

    // Open the file to write for complete gif
    char fname_complete[100] = "./simulation-files/fhn-";
    strcat(fname_complete, method);
    strcat(fname_complete, "-");
    strcat(fname_complete, s_dt);
    strcat(fname_complete, ".txt");
    FILE *fp_all = NULL;
    fp_all = fopen(fname_complete, "w");
    int save_rate = ceil(M / 100.0);

    // Open the file to write for times
    char fname_times[100] = "./simulation-files/sim-times-";
    strcat(fname_times, method);
    strcat(fname_times, "-");
    strcat(fname_times, s_dt);
    strcat(fname_times, ".txt");
    FILE *fp_times = NULL;
    fp_times = fopen(fname_times, "w");

    // For velocity
    bool tag = true;
    double velocity = 0.0;
    
    // Start timer
    double start, finish, elapsed = 0.0;

    start = omp_get_wtime();

    // ADI first order
    if (strcmp(method, "ADI1") == 0)
    {
        #pragma omp parallel num_threads(num_threads) default(none) \
        private(i, j, I_stim, dv_dt, dw_dt, diff_term) \
        shared(v, w, N, M, dx, dy, dt, G, eta1, eta2, eta3, vth, vp, ga, chi, Cm, L, s1_x_limit, \
        stim_strength, t_s1_begin, stim_duration, x_lim, t_s2_begin, stim2_duration, \
        x_max, y_max, x_min, y_min, fp_all, fp_times, time, tag, save_rate, \
        v_tilde, w_tilde, phi, T, tstep, step, r_v, rightside, solution, velocity)
        {
            while (step < M)
            {
                // Get time step
                tstep = time[step];

                // Predict v_tilde and w_tilde with explicit method
                #pragma omp for collapse(2) nowait
                for (i = 1; i < N-1; i++)
                {
                    for (j = 1; j < N-1; j++)
                    {
                        // Stimulus 1
                        if (tstep >= t_s1_begin && tstep <= t_s1_begin + stim_duration && j <= x_lim)
                        {
                            I_stim = stim_strength;
                        }
                        // Stimulus 2
                        else if (tstep >= t_s2_begin && tstep <= t_s2_begin + stim2_duration && j >= x_min && j <= x_max && i >= y_min && i <= y_max)
                        {
                            I_stim = stim_strength;
                        }
                        else 
                        {
                            I_stim = 0.0;
                        }

                        // Get dv_dt and dw_dt
                        dv_dt = reaction_v(v[i][j], w[i][j]) + I_stim;
                        dw_dt = reaction_w(v[i][j], w[i][j]);
                        
                        // Update v and w
                        v_tilde[i][j] = v[i][j] + dt * dv_dt;
                        w[i][j] = w[i][j] + dt * dw_dt;

                        // Update rightside for Thomas algorithm
                        rightside[j][i] = v_tilde[i][j];
                    }
                }
                
                // Diffusion
                // 1st: Implicit y-axis diffusion (lines)
                #pragma omp barrier
                #pragma omp for nowait
                for (i = 1; i < N-1; i++)
                {
                    thomas_algorithm(rightside[i], solution[i], N-2, phi);

                    // Update v
                    for (j = 1; j < N-1; j++)
                    {
                        v_tilde[j][i] = solution[i][j];
                    }
                }

                // 2nd: Implicit x-axis diffusion (columns)
                #pragma omp barrier
                #pragma omp for nowait
                for (i = 1; i < N-1; i++)
                {
                    thomas_algorithm(v_tilde[i], v[i], N-2, phi);
                }

                // Boundary conditions
                #pragma omp for nowait
                for (i = 0; i < N; i++)
                {
                    v[i][0] = v[i][1];
                    v[i][N-1] = v[i][N-2];
                    v[0][i] = v[1][i];
                    v[N-1][i] = v[N-2][i];
                }

                // Save data to file
                #pragma omp master
                {
                    // Write to file
                    if (step % save_rate == 0)
                    {
                        for (int i = 0; i < N; i++)
                        {
                            for (int j = 0; j < N; j++)
                            {
                                fprintf(fp_all, "%lf\n", v[i][j]);
                            }
                        }
                        fprintf(fp_times, "%lf\n", time[step]);
                    }

                    // Check S1 velocity
                    if (v[0][N-1] > 80 && tag)
                    {
                        velocity = ((10*(L - s1_x_limit)) / (time[step]));
                        printf("S1 velocity: %lf\n", velocity);
                        tag = false;
                    }
                }
                
                // Update step
                #pragma omp master
                {
                    step++;
                }
                #pragma omp barrier 

            }
        }
    }

    // ADI second order
    else if (strcmp(method, "ADI2") == 0)
    {
        #pragma omp parallel num_threads(num_threads) default(none) \
        private(i, j, I_stim, dv_dt, dw_dt, diff_term) \
        shared(v, w, N, M, dx, dy, dt, G, eta1, eta2, eta3, vth, vp, ga, chi, Cm, L, s1_x_limit, \
        stim_strength, t_s1_begin, stim_duration, x_lim, t_s2_begin, stim2_duration, \
        x_max, y_max, x_min, y_min, fp_all, fp_times, time, tag, save_rate, \
        v_tilde, w_tilde, phi, T, tstep, step, r_v, rightside, solution, velocity)
        {
            while (step < M)
            {
                // Get time step
                tstep = time[step];

                // Predict v_tilde and w_tilde with explicit method
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        // Stimulus 1
                        if (tstep >= t_s1_begin && tstep <= t_s1_begin + stim_duration && j <= x_lim)
                        {
                            I_stim = stim_strength;
                        }
                        // Stimulus 2
                        else if (tstep >= t_s2_begin && tstep <= t_s2_begin + stim2_duration && j >= x_min && j <= x_max && i >= y_min && i <= y_max)
                        {
                            I_stim = stim_strength;
                        }
                        else 
                        {
                            I_stim = 0.0;
                        }

                        // Get dv_dt and dw_dt
                        dv_dt = reaction_v(v[i][j], w[i][j]) + I_stim;
                        dw_dt = reaction_w(v[i][j], w[i][j]);
                        
                        // Update v_tilde and w_tilde
                        v_tilde[i][j] = v[i][j] + ((phi * 0.5) * (diffusion_i_2nd(i, j, N, v) + diffusion_j_2nd(i, j, N, v))) + (dt * 0.5 * dv_dt);
                        w_tilde[i][j] = w[i][j] + (dt * 0.5) * dw_dt;

                        // Update r_v for Thomas algorithm
                        dv_dt = reaction_v(v_tilde[i][j], w_tilde[i][j]) + I_stim;
                        r_v[i][j] = 0.5 * dt * dv_dt;
                        
                        // Update w
                        dw_dt = reaction_w(v_tilde[i][j], w_tilde[i][j]);
                        w[i][j] = w[i][j] + dt * dw_dt;
                    }
                }
                
                // Update rightside for Thomas algorithm
                #pragma omp barrier
                // 1st: Implicit y-axis diffusion (lines): right side with explicit x-axis diffusion (columns)
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        // Explicit diffusion in x-axis
                        diff_term = v[i][j] + (phi*0.5) * diffusion_j_2nd(i, j, N, v);
                        
                        // Update rightside
                        rightside[j][i] = diff_term + r_v[i][j];
                    }
                }

                // Solve tridiagonal system for v
                #pragma omp for nowait
                for (i = 0; i < N; i++)
                {
                    thomas_algorithm_2nd(rightside[i], solution[i], N, (phi*0.5));
                    
                    // Update v
                    for (j = 0; j < N; j++)
                    {
                        v[j][i] = solution[i][j];
                    }
                }
                
                // Update rightside for Thomas algorithm
                #pragma omp barrier
                // 2nd: Implicit x-axis diffusion (columns): right side with explicit y-axis diffusion (lines)
                #pragma omp for collapse(2)
                for (i = 0; i < N; i++)
                {
                    for (j = 0; j < N; j++)
                    {
                        // Explicit diffusion in y-axis
                        diff_term = v[i][j] + (phi*0.5) * diffusion_i_2nd(i, j, N, v);;
                        
                        // Update rightside
                        rightside[i][j] = diff_term + r_v[i][j];
                    }
                }
                
                // Solve tridiagonal system for v
                #pragma omp for nowait
                for (i = 0; i < N; i++)
                {
                    thomas_algorithm_2nd(rightside[i], v[i], N, (phi*0.5));
                }

                // Save data to file
                #pragma omp master
                {
                    // Write to file
                    if (step % save_rate == 0)
                    {
                        for (int i = 0; i < N; i++)
                        {
                            for (int j = 0; j < N; j++)
                            {
                                fprintf(fp_all, "%lf\n", v[i][j]);
                            }
                        }
                        fprintf(fp_times, "%lf\n", time[step]);
                    }

                    // Check S1 velocity
                    if (v[0][N-1] > 80 && tag)
                    {
                        velocity = ((10*(L - s1_x_limit)) / (time[step]));
                        printf("S1 velocity: %lf\n", velocity);
                        tag = false;
                    }
                }
                
                // Update step
                #pragma omp master
                {
                    step++;
                }
                #pragma omp barrier 

            }
        }
    }

    // Forward Euler
    else if (strcmp(method, "FE") == 0)
    {
        #pragma omp parallel num_threads(num_threads) default(none) \
        private(i, j, I_stim, dv_dt, dw_dt, diff_term) \
        shared(v, w, N, M, dx, dy, dt, G, eta1, eta2, eta3, vth, vp, ga, chi, Cm, L, s1_x_limit, \
        stim_strength, t_s1_begin, stim_duration, x_lim, t_s2_begin, stim2_duration, \
        x_max, y_max, x_min, y_min, fp_all, fp_times, time, tag, save_rate, \
        v_tilde, w_tilde, phi, T, tstep, step, r_v, rightside, solution, velocity)
        {
            while (step < M)
            {
                // Get time step
                tstep = time[step];

                // Predict v_tilde and w_tilde with explicit method
                #pragma omp for collapse(2)
                for (i = 1; i < N-1; i++)
                {
                    for (j = 1; j < N-1; j++)
                    {
                        // Stimulus 1
                        if (tstep >= t_s1_begin && tstep <= t_s1_begin + stim_duration && j <= x_lim)
                        {
                            I_stim = stim_strength;
                        }
                        // Stimulus 2
                        else if (tstep >= t_s2_begin && tstep <= t_s2_begin + stim2_duration && j >= x_min && j <= x_max && i >= y_min && i <= y_max)
                        {
                            I_stim = stim_strength;
                        }
                        else 
                        {
                            I_stim = 0.0;
                        }

                        // Get dv_dt and dw_dt
                        dv_dt = reaction_v(v[i][j], w[i][j]) + I_stim;
                        dw_dt = reaction_w(v[i][j], w[i][j]);
                        
                        // Update v and w
                        v_tilde[i][j] = v[i][j] + dt * dv_dt;
                        w[i][j] = w[i][j] + dt * dw_dt;
                    }
                }

                // Boundary conditions
                #pragma omp for
                for (i = 0; i < N; i++)
                {
                    v_tilde[i][0] = v_tilde[i][1];
                    v_tilde[i][N-1] = v_tilde[i][N-2];
                    v_tilde[0][i] = v_tilde[1][i];
                    v_tilde[N-1][i] = v_tilde[N-2][i];
                }
                
                // Diffusion
                #pragma omp barrier
                #pragma omp for collapse(2)
                for (i = 1; i < N-1; i++)
                {
                    for (j = 1; j < N-1; j++)
                    {   
                        v[i][j] = v_tilde[i][j] + phi * (v_tilde[i][j-1] - 2.0*v_tilde[i][j] + v_tilde[i][j+1] + v_tilde[i-1][j] - 2.0*v_tilde[i][j] + v_tilde[i+1][j]);
                    }
                }

                // Boundary conditions
                #pragma omp for nowait
                for (i = 0; i < N; i++)
                {
                    v[i][0] = v[i][1];
                    v[i][N-1] = v[i][N-2];
                    v[0][i] = v[1][i];
                    v[N-1][i] = v[N-2][i];
                }

                // Save data to file
                #pragma omp master
                {
                    // Write to file
                    if (step % save_rate == 0)
                    {
                        for (int i = 0; i < N; i++)
                        {
                            for (int j = 0; j < N; j++)
                            {
                                fprintf(fp_all, "%lf\n", v[i][j]);
                            }
                        }
                        fprintf(fp_times, "%lf\n", time[step]);
                    }

                    // Check S1 velocity
                    if (v[0][N-1] > 80 && tag)
                    {
                        velocity = ((10*(L - s1_x_limit)) / (time[step]));
                        printf("S1 velocity: %lf\n", velocity);
                        tag = false;
                    }
                }
                
                // Update step
                #pragma omp master
                {
                    step++;
                }
                #pragma omp barrier 

            }
        }
    }

    // Check time
    finish = omp_get_wtime();
    elapsed = finish - start;

    printf("\nElapsed time = %e seconds\n", elapsed);

    // Comparison file
    FILE *fp_comp = NULL;
    fp_comp = fopen("comparison.txt", "a");
    fprintf(fp_comp, "%s  \t|\t%d threads\t|\t%.3f ms\t|\t%lf m/s\t|\t%e seconds\n", method, num_threads, dt, velocity, elapsed);

    // Close files
    fclose(fp_all);
    fclose(fp_times);
    fclose(fp_comp);
    
    // Free alocated memory
    free(time);
    free(v);
    free(w);
    free(v_tilde);
    free(w_tilde);
    free(r_v);
    free(rightside);
    free(solution);

    return 0;
}