clc;
clear;
aa = dir('*.edf');
filename = aa.name;
[hdr, xx] = edfread(filename);
[data, header, cfg] = lab_read_edf(filename);
clear data;
%fs_original = ceil(hdr.frequency(1));
fs_original = round(header.hdr.numbersperrecord(1) / header.hdr.duration);
[p, f, ~] = fileparts(filename);
patient_name = f;
channel_names = string(cellstr(header.hdr.channelname));
channel_names = strrep(channel_names, " ", "");
ECG_final =(xx(29,:));

% Notch filter at 50 Hz
f0 = 50;
Q = 30;
[b, a] = iirnotch(f0 / (fs_original / 2), f0 / (fs_original * Q));
ECG_final = filtfilt(b, a, ECG_final);

% High-pass filter at 0.5 Hz
[b, a] = butter(4, .5 / (fs_original/2),'high');
ECG_final = filtfilt(b, a, ECG_final);

% Low-pass filter at 20 Hz
[b, a] = butter(4, 20 / (fs_original / 2), 'low');
ECG_final = filtfilt(b, a, ECG_final);

% --- DOWNSAMPLE from fs_original to 250 Hz ---
fs = 250; % Target frequency
[ecg_down, ~] = resample(ECG_final, fs, fs_original);

% --- Baseline removal ---
baseline_shift = mean(ECG_final(1:5000));
ECG_final = ECG_final - baseline_shift;

timing_table = readtable('Timings.xlsx', 'Sheet', 'Sheet1');
onset_cell = timing_table{1, 2};

if iscell(onset_cell)
    onset_str = onset_cell{1};
else
    onset_str = onset_cell;
end

%Remove double quotes
onset_str = strrep(char(onset_str), '"', '');

disp("Parsed onset string:");
disp(onset_str);

tOnset = sscanf(onset_str, '%d:%d:%d');

if numel(tOnset) ~= 3
    error("Onset time format is incorrect.");
end

%if numel(tOnset) ~= 3, error("Onset time format is incorrect."); end
onset_in_seconds = tOnset(1)*3600 + tOnset(2)*60 + tOnset(3);

tStart = sscanf(strrep(hdr.starttime, '.', ':'), '%d:%d:%d');
if numel(tStart) ~= 3, error("Start time format is incorrect."); end

start_in_seconds = tStart(1)*3600 + tStart(2)*60 + tStart(3);

%Condition for onset time less than start of filename
if onset_in_seconds < start_in_seconds
    actual_onset_in_seconds = (24*3600 - start_in_seconds) + onset_in_seconds;
else
    actual_onset_in_seconds = onset_in_seconds - start_in_seconds;
end

preictal_start_time = actual_onset_in_seconds-(15*60);
preictal_end_time = actual_onset_in_seconds ;
interictal_start_time = actual_onset_in_seconds+(20*60);
interictal_end_time =actual_onset_in_seconds+(35*60);

% Ensure sample indices are within bounds
max_index = length(ECG_final);

start_pre = preictal_start_time * fs;
end_pre = preictal_end_time * fs;
start_inter = interictal_start_time * fs;
end_inter = interictal_end_time * fs;

% Check if indices exceed the ECG length
if end_pre > max_index
    warning('Preictal end index exceeds signal length. Truncating.');
    end_pre = max_index;
end

if end_inter > max_index
    warning('Interictal end index exceeds signal length. Truncating.');
    end_inter = max_index;
end

ECG_final_preictal = ECG_final(start_pre:end_pre);
ECG_final_interictal = ECG_final(start_inter:end_inter);

samples_per_chunk = 10 * fs;
num_chunks = 90;
required_samples = num_chunks * samples_per_chunk;

if length(ECG_final_preictal) < required_samples || length(ECG_final_interictal)< required_samples
    error("Not enough ECG data for the required chunking.");
end

ECG_final_preictal = ECG_final_preictal(1:required_samples);
ECG_final_interictal = ECG_final_interictal(1:required_samples);

matrix_preictal = reshape(ECG_final_preictal, samples_per_chunk, num_chunks)';
matrix_interictal = reshape(ECG_final_interictal, samples_per_chunk, num_chunks)';

save('data.mat','matrix_interictal','matrix_preictal');

% Add labels
matrix_preictal = [matrix_preictal, ones(num_chunks, 1)];
matrix_interictal = [matrix_interictal, zeros(num_chunks, 1)];

writematrix(matrix_preictal, 'ECG_chunks_preictal_file_1.xlsx');
writematrix(matrix_interictal, 'ECG_chunks_interictal_file_1.xlsx');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compute HRV
feature_names = {'BPMglobal', 'BPMlocal', 'SDNN', 'rmsRR', 'NN50', ...
'pNN50', 'IBImean', 'IBIrange_1', 'IBIrange_2', ...
'SD1', 'SD2', 'lfPower', 'hfPower'};

num_features = numel(feature_names);

hrv_matrix_preictal = zeros(num_chunks, num_features);
hrv_matrix_interictal = zeros(num_chunks, num_features);

%preictal
%matrix_preictal = matrix_preictal/10;

for i = 1:num_chunks
    figure;
    plot((0:samples_per_chunk-1)/fs, matrix_preictal(i,1:end-1)); % exclude label column
    title(sprintf('Preictal ECG Chunk %d', i));
    xlabel('Time (s)');
    ylabel('Amplitude');

    i

    d = ECG(matrix_preictal(i,1:end-1), fs, 100);
    d.init;

    hrv_matrix_preictal(i,:) = [d.BPMglobal, d.BPMlocal, d.SDNN, d.rmsRR, ...
        d.NN50, d.pNN50, d.IBImean, d.IBIrange(1), d.IBIrange(2), ...
        d.SD1, d.SD2, d.lfPower, d.hfPower];
end

% Add label column to preictal HRV features
hrv_matrix_preictal = [hrv_matrix_preictal, ones(num_chunks, 1)];

feature_names_with_label = [feature_names, {'Label'}];

xlswrite('HRV_features_preictal.xlsx', feature_names_with_label, 'Sheet1', 'A1');
xlswrite('HRV_features_preictal.xlsx', hrv_matrix_preictal, 'Sheet1', 'A2');

% interictal
%matrix_interictal = round(matrix_interictal / 10);

for i = 1:num_chunks
    figure;
    plot((0:samples_per_chunk-1)/fs, matrix_interictal(i,1:end-1)); % exclude label column
    title(sprintf('Interictal ECG Chunk %d', i));
    xlabel('Time (s)');
    ylabel('Amplitude');

    d = ECG(matrix_interictal(i,1:end-1), fs, 100);
    d.init;

    hrv_matrix_interictal(i,:) = [d.BPMglobal, d.BPMlocal, d.SDNN, d.rmsRR, ...
        d.NN50, d.pNN50, d.IBImean, d.IBIrange(1), d.IBIrange(2), ...
        d.SD1, d.SD2, d.lfPower, d.hfPower];
end

% Add label column to interictal HRV features
hrv_matrix_interictal = [hrv_matrix_interictal, zeros(num_chunks, 1)];

xlswrite('HRV_features_interictal.xlsx', feature_names_with_label, 'Sheet1', 'A1');
xlswrite('HRV_features_interictal.xlsx', hrv_matrix_interictal, 'Sheet1', 'A2');

save('hrv_features.mat', 'matrix_interictal', 'matrix_preictal', ...
    'hrv_matrix_preictal', 'hrv_matrix_interictal');

disp('All ECG and HRV data processed and saved.');