clear all
close all
clc

function data = readDataFile(filename)
    fileID = fopen(filename, 'r');
    
    if fileID == -1
        error('Could not open file: %s', filename);
    end
    data = {};
    try
        line = fgetl(fileID);
        while ischar(line)
            values = sscanf(line, '%f');
            
            if ~isempty(values)
                data{end+1} = values;
            end
            
            line = fgetl(fileID);
        end
        
        if all(cellfun(@length, data) == length(data{1}))
            data = cell2mat(data);
        end
        
    catch ME
        fclose(fileID);
        rethrow(ME);
    end
    fclose(fileID);
end

app_disp = readDataFile('GRF_applied.txt');
sensor_loc_disp = readDataFile('sensor_loc.txt');
coord_x = readDataFile('coord_x.txt');
coord_y = readDataFile('coord_y.txt');
disp_x = readDataFile('disp_x.txt');
disp_y = readDataFile('disp_y.txt');

save('../cantilever_beam_deflection.mat','coord_x','coord_y','disp_x',...
    'disp_y','app_disp','sensor_loc_disp');
