
% you need to first run csv2mat_all_files
% csv2mat_all_files : files for the changing csv format to Matlab 


lfp_changeSetup('gun') ;
lfp_getEvtIDs;
lfp_declareGlobals;

green_target ='PNOC';
% run this as well after PNOC 
% green_target ='NTS';

path_name = sprintf('%s_30hz_updated_final', green_target);
ag_read2(path_name)

% [1 3] is for DMS
% [2 4] is for DLS 
hf1 = lfp_makepasteup({TrialStart InitOff TurnEnd [LOn ROn] [LOff ROff]}, @lfp_disp, [], [1,3], [], 'avg','err2', 'pasteup_opts', {'autoselect'});
% if we are using the figure from matlba, you can just set  hf1 =1
hl = findobj(hf1,'Type', 'line');

for idx = 1:length(hl)
    if mod(idx, 8) == 0
        set(hl(idx), 'Color', [0, 0, 1]);
    elseif mod(idx, 8) == 7 || mod(idx, 8) == 6
        set(hl(idx), 'Color', [0, 0, 0.8], LineWidth=0.75);
    elseif mod(idx,8)== 5
        set(hl(idx), 'Color', [1, 0, 0]);
    elseif mod(idx, 8) == 4 || mod(idx, 8) == 3
        set(hl(idx), 'Color', [0.8, 0, 0], LineWidth=0.75);
    else
        set(hl(idx), 'Color', 'black',LineWidth=1, linestyle =':' )
    end
    
end


title('');
titleHandle = get(gca, 'Title'); % Get handle to title object
delete(titleHandle); 
% hTitle = title('Activity of Dopamine and Astrocyte', 'FontSize', 14); % Get the handle of the current axes title
% titlePos = get(hTitle, 'Position'); % Get the current position of the title
% newPos = titlePos + [0, 0.05,0]; % Adjust the position by dx, dy, dz as needed
% set(hTitle, 'Position', newPos, 'HorizontalAlignment', 'center'); % Set the new position and alignment
% Find all text objects in the figure
textObjects = findall(gca, 'Type', 'text');
% Loop through all text objects and change the string if it matches 'evt[11,12]' or 'evt[4,6]'
for k = 1:length(textObjects)
    if strcmp(get(textObjects(k), 'String'), 'evt[3 5]')
        set(textObjects(k), 'String', 'SideEnt',  'Rotation', 90); % Replace with your new text
    elseif strcmp(get(textObjects(k), 'String'), 'evt[4 6]')
        set(textObjects(k), 'String', 'SideExit' ,'Rotation', 90); % Replace with your new text
        elseif strcmp(get(textObjects(k), 'String'), 'evt[11 12]')
        set(textObjects(k), 'String', 'SideExit' ,'Rotation', 90);
    
    elseif strcmp(get(textObjects(k), 'String'), 'InitOff')
        set(textObjects(k), 'String', 'InitExit' ,'Rotation', 90);
    elseif strcmp(get(textObjects(k), 'String'), 'TrialStart')
        set(textObjects(k), 'String', 'TrialStart' ,'Rotation', 90);
    elseif strcmp(get(textObjects(k), 'String'), 'TurnEnd')
        set(textObjects(k), 'String', 'Turn' ,'Rotation', 90);

    end
end

% set(hl(1), 'Color', 'black', LineWidth=1, linestyle =':');
% set(hl(2), 'Color', 'black', LineWidth=1, linestyle =':');
% set(hl(3), 'Color', 'black', LineWidth=1, linestyle =':');
% 
% 
% set(hl(13), 'Color', 'black', LineWidth=1, linestyle =':');
% set(hl(25), 'Color', 'black', LineWidth=1, linestyle =':');
% set(hl(37), 'Color', 'black', LineWidth=1, linestyle =':');
% set(hl(49), 'Color', 'black', LineWidth=1, linestyle =':');
side_port_entry_x = get(hl(13), 'XData');
side_port_entry_y = get(hl(13), 'yData');

% Add the new line to the figure
% Airpuff happens 200ms after side port entry
% Plotting invisible data points for the legend
hold on; % Retain current plot
% newLineHandle = plot(side_port_entry_x + 0.2, side_port_entry_y, 'Color', [0.8 0.8 0.8], 'LineWidth', 2);
% text(side_port_entry_x(1) + 0.2, 1.3, "Reward",'Fontsize',20);
hLegendAstrocyte = plot(NaN, NaN, 'green', LineWidth=1); % Plot an invisible point for the legend
hLegendDopamine = plot(NaN, NaN, 'red', LineWidth=1); % Plot an invisible point for the legend
legend([hLegendAstrocyte, hLegendDopamine], 'S-D1', 'Dopamine', 'Location', 'southeast');
hold off; % Release the hold on the current plot

% Convert mm to inches for width and height (1 inch = 25.4 mm)


grid off;  % This turns off all grid lines
set(gca, 'box', 'off');
ax = gca;
ax.Box = 'off';
% Set x-axis limits and ticks
ax.XTick = 0:1:3;                  % Set x-axis ticks to occur every 1 second

% Set y-axis limits and ticks
% ax.YLim = [-1,1.5];               % Set y-axis limits to match your data range
ax.YTick = -0.5:0.5:1;             % Set y-axis ticks with an interval that includes 0.2 dF/F

xlabel('Time(s)');
ylabel('Z-score');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 14);
set(gca, 'FontName', 'Arial')
figWidth = 8; % Width in inches
figHeight = 6; % Height in inches

% Convert figure size to centimeters if needed (1 inch = 2.54 cm)
figWidth_cm = figWidth * 2.54;
figHeight_cm = figHeight * 2.54;

% Set the size of the figure
set(gcf, 'Units', 'Inches', 'Position', [0, 0, figWidth, figHeight]);
saveas(hf1,sprintf('%s_neuron_dopamine_DMS_new.pdf', green_target));
saveas(hf1,sprintf('%s_neuron_dopamine_DMS_new.fig', green_target));