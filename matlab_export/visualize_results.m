% visualize_results.m
% Run this script in MATLAB from inside the "matlab_export" folder.

clear; clc; close all;

%% 1. Visualize Training Progress
fprintf('Loading training_progress.csv...\n');
progress = readtable('training_progress.csv');

figure('Name', 'Co-Training Progress', 'Position', [100, 200, 800, 450]);
% Sat 1 (Red)
plot(progress.steps_completed, progress.sat1_reward_mean, 'r-', 'LineWidth', 2, 'DisplayName', 'Sat 1 (Red)');
hold on;
% Sat 2 (Orange) - MATLAB uses RGB [1, 0.5, 0] for orange
plot(progress.steps_completed, progress.sat2_reward_mean, '-', 'Color', [1, 0.5, 0], 'LineWidth', 2, 'DisplayName', 'Sat 2 (Orange)');

grid on;
title('Co-Training Progress (100,000 steps)');
xlabel('Training Steps Completed');
ylabel('Mean Episodic Reward');
legend('Location', 'northwest');
set(gca, 'FontSize', 12);


%% 2. Visualize Final Evaluation Episode
fprintf('Loading reward_history.csv...\n');
eval_data = readtable('reward_history.csv');

figure('Name', 'Evaluation Episode', 'Position', [150, 150, 800, 450]);

% --- Subplot 1: Reward per step ---
subplot(2, 1, 1);
plot(eval_data.step, eval_data.sat1_reward, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Sat 1');
hold on;
plot(eval_data.step, eval_data.sat2_reward, '-x', 'Color', [1, 0.5, 0], 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'Sat 2');
grid on;
title('Evaluation: Immediate Reward per Step');
ylabel('Reward');
legend('Location', 'northeast');
set(gca, 'FontSize', 11);

% --- Subplot 2: Cumulative Reward ---
subplot(2, 1, 2);
plot(eval_data.step, eval_data.sat1_cumulative_reward, 'r-', 'LineWidth', 2, 'DisplayName', 'Sat 1');
hold on;
plot(eval_data.step, eval_data.sat2_cumulative_reward, '-', 'Color', [1, 0.5, 0], 'LineWidth', 2, 'DisplayName', 'Sat 2');
grid on;
title('Evaluation: Cumulative Reward Over Time');
xlabel('Step');
ylabel('Total Reward');
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

fprintf('Done! Figures should now be displayed.\n');
