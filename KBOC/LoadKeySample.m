function ks = LoadKeySample(file)
% LOADKEYSAMPLE(FILE) loads a keystroking file from the BIOSECUR-KEYS
% database.  The output is a structure with the fields:
%
%   'num_strokes'        Total number of strokes in the sample.
%
%   'keys'               Scan code of the keys that where pressed or
%                        released during the acquisition.
%
%   'mask'               When overlapped with the field 'keys', ones
%                        indicate key press events and zeros mean key
%                        release events.
%
%   'timing'             When overlapped with the fields 'keys', this
%                        field contains the number of milliseconds since
%                        the last key press or release.
%
%   'labels'             Labels of the keys that where pressed during the
%                        acquisition. Key release events are ignored.
%
%   'text'               Textual transcription of the text written in the
%                        acquisition (like 'labels', but control keys are
%                        ignored).

scancode_mat = {
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...        % 1-10
    ' '  ' '  ' '  ' '  ' '  'Q'  'W'  'E'  'R'  'T'  ...        % 11-20
    'Y'  'U'  'I'  'O'  'P'  ' '  ' '  ' '  ' '  'A'  ...        % 21-30
    'S'  'D'  'F'  'G'  'H'  'J'  'K'  'L'  'Ñ'  'TILDE'  ...        % 31-40
    ' '  'LSHIFT'  ' '  'Z'  'X'  'C'  'V'  'B'  'N'  'M'  ...        % 41-50
    ' '  '.'  ' '  'RSHIFT'  ' '  ' '  '_'  'CAPSLOCK'  ''  ' '  ...        % 51-60
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...        % 61-70
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...        % 71-80
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
    ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ' '  ...
	       };
    

[fid] = fopen(file, 'r');
if fid == -1
    error(['Couldn''t open file ''' file '''']);
end

ks = struct('num_strokes', [], 'labels', [], 'keys', [], 'mask', [], ...
	    'timing', [], 'text', []);

% Read the file name
tmp = strfind(file, '\');
filename = file(tmp(end)+1 : end);

tline = fgetl(fid);
if feof(fid)
    fprintf(1, [filename '  (Empty file)\n']);
    fclose(fid);
    return;
end

val = str2num(tline);
if ~isempty(val)
    ks.num_strokes = val;
else
    fprintf(1, [filename '  (Wrong syntax in line ''' tline ''')\n']);
    fclose(fid);
    return;
end

while ~feof(fid)
    tline = strrep(fgetl(fid),'D','J');  % get next line, changing Ds with Js to avoid ambiguity when parsing hexadecimals
    val = sscanf(tline, '%x%c%d');
    if length(val) < 1  % empty line
	continue;
    end
    
    ks.keys = [ks.keys val(1)];
    ks.mask = [ks.mask  (char(val(2)) == 'J')];  % 1 if key down, 0 if key up
    ks.timing = [ks.timing  val(3)];
end

for i = find(ks.mask)
    ks.labels = [ks.labels  scancode_mat(ks.keys(i))];
end

for i = 1:length(ks.labels)
    if ~ strcmp(ks.labels(i),'RSHIFT') & ~ strcmp(ks.labels(i),'LSHIFT') ...
	    & ~ strcmp(ks.labels(i),'TILDE') & ~ strcmp(ks.labels(i), ...
							'CAPSLOCK')
	ks.text = strcat(ks.text, ks.labels(i));
    end

end

fclose(fid);