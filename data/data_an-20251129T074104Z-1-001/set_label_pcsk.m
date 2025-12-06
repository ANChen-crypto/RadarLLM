%% ===== 0) 路徑與參數（自行修改） =====
clc; clear;
MMWAVE_DIR = "D:\deff\0925永康1\f1";              % 內含 replay_*.json
KINECT_CSV = "D:\deff\0925永康1\f1\output.csv";  % output.csv 路徑
save_fir= MMWAVE_DIR;
POINT_KEY  = "pointCloud";     % 你的點雲欄位名
MAX_GAP_MS = 0.08;             % 對齊容許（秒）
POINT_SIZE = 40;               % 點雲大小
mAX = [-2 2 0 4 -1 2];         % mmWave 顯示範圍
sAX = [-1 1 -1 1 -1 1]*0.5;        % Skeleton 顯示範圍
FPS = 60;                      % 播放/影片幀率

OUT_MP4 = fullfile(save_fir,"labels_playback.mp4");
OUT_CSV = fullfile(save_fir,"aligned_with_labels.csv");
% 骨架設定
sk_index = [0,11,12,13,14,15,16,23,24,25,26,27,28];
caserh = [1,2,4,6]; caselh = [1,3,5,7]; caserl = [2,8,10,12]; casell = [3,9,11,13];

%% ===== 1) 讀 mmWave =====
frames = [];
jfs = dir(fullfile(MMWAVE_DIR,"replay_*.json"));
assert(~isempty(jfs),"在 %s 找不到 replay_*.json", MMWAVE_DIR);
nums = zeros(numel(jfs),1);
for i = 1:numel(jfs)
    m = regexp(jfs(i).name,"^replay_(\d+)\.json$","tokens","once");
    if ~isempty(m), nums(i) = str2double(m{1}); end
end
[~,ord] = sort(nums); jfs = jfs(ord);
for i = 1:numel(jfs)
    d = jsondecode(fileread(fullfile(jfs(i).folder,jfs(i).name)));
    if ~isfield(d,'data'), continue; end
    if isstruct(d.data), frames = [frames; d.data(:)];
    elseif iscell(d.data), frames = [frames; [d.data{:}]'];
    end
end
assert(~isempty(frames),"replay_* 檔案裡沒有 data 幀");

hh = zeros(numel(frames),1); mm = hh; ss = hh; mm_ts_s = hh;
for i = 1:numel(frames)
    ts = string(frames(i).timestamp);
    dt = datetime(ts,'InputFormat','yyyy/MM/dd HH:mm:ss.SSS');
    hh(i) = hour(dt); mm(i) = minute(dt); ss(i) = second(dt);
    mm_ts_s(i) = hh(i)*3600 + mm(i)*60 + ss(i);
end
[mm_ts_s, ord] = sort(mm_ts_s);
frames = frames(ord); hh = hh(ord); mm = mm(ord); ss = ss(ord);

%% ===== 2) 讀 Kinect =====
kinect = table2array(readtable(KINECT_CSV));
sktime  = kinect(:,1:4);  skdata  = kinect(:,5:end);
H=sktime(:,1); M=sktime(:,2); S=sktime(:,3); MS=sktime(:,4);
sk_ts_s = H*3600 + M*60 + S + MS*0.001;
select_skdata = cell(size(skdata,1),1);
for i=1:size(skdata,1)
    row = skdata(i,:); tmp = zeros(numel(sk_index),3);
    for k=1:numel(sk_index)
        j = sk_index(k);
        tmp(k,1) = row(4*j+1) - 0.5;
        tmp(k,2) = row(4*j+2) - 0.5;
        tmp(k,3) = row(4*j+3);
    end
    select_skdata{i} = tmp;
end

%% ===== 3) 對齊 =====
N = numel(mm_ts_s);
aligned = nan(N,7);   % [mm_idx, sk_idx, hh, mm, ss, diff_s, label]
for i = 1:N
    [gap, j] = min(abs(sk_ts_s - mm_ts_s(i)));
    aligned(i,1:6) = [i, j, hh(i), mm(i), ss(i), mm_ts_s(i)-sk_ts_s(j)];
    aligned(i,7) = 0;   % 預設 label=0
end

%% ===== 4) 播放 & 標註 =====
frameDur = 1/FPS;
f = figure('Color','w','Name','Labeler: 按 1~9 標註, q=退出');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
v = VideoWriter(OUT_MP4, 'MPEG-4'); v.FrameRate = FPS; open(v);

currentLabel = 0; quitFlag = false;
set(f,'WindowKeyPressFcn', @onKey);
for i = 1:length(aligned)
    if getappdata(f,'quit'), break; end

    % ---- 取 mmWave 點雲 ----
    PC = [];
    F = frames(i); if isfield(F,'frameData'), F = F.frameData; end
    if isfield(F,POINT_KEY)
        A = F.(POINT_KEY);
        if isnumeric(A), PC=A;
        elseif isstruct(A)&&all(isfield(A,{'x','y','z'})), PC=[[A.x].',[A.y].',[A.z].'];
        elseif iscell(A), PC=cell2mat(cellfun(@(x)x(:).',A,'UniformOutput',false));
        end
        if ~isempty(PC) && size(PC,2)>=3, PC=PC(:,1:3); else, PC=[]; end
    end

    % ---- 上：mmWave ----
    nexttile(1); cla;
    if ~isempty(PC)
        scatter3(PC(:,1),PC(:,2),PC(:,3),POINT_SIZE,'r','filled');  % 實心大點
    end
    axis(mAX); grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
    set(gca,'ZDir','reverse'); view(0,180);
    title(sprintf('mmWave  |  %02d:%02d:%06.3f  (frame %d)', ...
        round(aligned(i,3)), round(aligned(i,4)), aligned(i,5), i));

    % ---- 下：Skeleton ----
    nexttile(2); cla; j = aligned(i,2);
    if ~isnan(j) && abs(aligned(i,6)) <= MAX_GAP_MS
        aa = select_skdata{j};
        t = aa(caserh,:); plot3(t(:,1)',t(:,3)',t(:,2)','-o','MarkerFaceColor','r'); hold on;
        t = aa(caselh,:); plot3(t(:,1)',t(:,3)',t(:,2)','-o','MarkerFaceColor','r');
        t = aa(caserl,:); plot3(t(:,1)',t(:,3)',t(:,2)','-o','MarkerFaceColor','r');
        t = aa(casell,:); plot3(t(:,1)',t(:,3)',t(:,2)','-o','MarkerFaceColor','r');
        axis(sAX); grid on; view(0,-180);
        title(sprintf('Skeleton | matched idx %d  |  Δt = %+7.3f s', j, aligned(i,6)));
    else
        axis(sAX); grid on; view(0,-180);
        title(sprintf('Skeleton | no match (|Δt| > %.3fs)', MAX_GAP_MS));
    end

    % ---- 這段是關鍵：在一個幀期間持續監聽鍵盤，抓到就寫入 ----
    t0 = tic;
    wrote = false;
    while toc(t0) < frameDur
        drawnow limitrate;          % 讓 KeyPressFcn 有機會觸發
        if getappdata(f,'quit'), break; end
        lbl = getappdata(f,'label');
        if ~isempty(lbl) && lbl>=1 && lbl<=9
            aligned(i,7) = lbl;     % 寫入當前幀的標籤
            setappdata(f,'label',0);% 清除，避免帶到下一幀
            wrote = true;
        end
    end

    % 可選：把本幀標到的數字顯示在上方標題（視覺確認）
    if wrote
        nexttile(1);
        ttl = get(get(gca,'Title'),'String');
        title([ttl, sprintf('  |  label=%d', aligned(i,7))]);
        drawnow;
    end

    % 寫影片
    frm = getframe(f); writeVideo(v,frm);
end

close(v);


%% ===== 5) 後處理：補標籤 =====
[labels_fixed, n_filled] = fix_isolated_labels(aligned(:,7), 5);
aligned(:,7) = labels_fixed;
fprintf('補上 %d 幀的標籤（孤立 1~9 後延最多 7 幀）。\n', n_filled);
%% ===== 5) 輸出 CSV =====
T = array2table(aligned, 'VariableNames', ...
    {'mm_idx','sk_idx','hour','minute','second','delta_t_s','label'});
writetable(T, OUT_CSV);
fprintf('影片存成 %s\nCSV 存成 %s\n', OUT_MP4, OUT_CSV);
%% ===== local function =====
function onKey(src, evt)
    % src = figure handle；狀態放在 figure appdata 裡
    key = evt.Key;

    % 退出
    if any(strcmp(key,{'q','escape'}))
        setappdata(src,'quit',true);
        return;
    end

    % 1~9（主鍵盤或數字鍵）
    d = NaN;
    if strlength(key)==1 && isstrprop(key,'digit')
        d = str2double(key);                            % '1'..'9'
    elseif startsWith(key,'numpad')
        d = str2double(extractAfter(key,'numpad'));     % 'numpad1'..'numpad9'
    end
    if ~isnan(d) && d>=1 && d<=9
        setappdata(src,'label',d);
    end
end
%%
function [ml_fixed, n_filled] = fix_isolated_labels(ml, lookahead)
% 若某幀為 1~9，且前後皆為 0，則把其後連續 0 的最多 lookahead 幀補成該標籤
% 不會覆蓋已存在的非 0 標籤；遇到非 0 便停止補標
    if nargin < 2, lookahead = 7; end
    N = numel(ml);
    n_filled = 0;
    for i = 1:N
        k = ml(i);
        if k >= 1 && k <= 9
            prevZero = (i == 1)  || (ml(i-1) == 0);
            nextZero = (i == N)  || (ml(i+1) == 0);
            if prevZero && nextZero
                jEnd = min(i + lookahead, N);
                for j = i+1 : jEnd
                    if ml(j) == 0
                        ml(j) = k;
                        n_filled = n_filled + 1;
                    else
                        break;  % 後面遇到非 0 就不再延伸
                    end
                end
            end
        end
    end
    ml_fixed = ml;
end
