function data = read_npy(npy_path)
%READ_NPY  Read a NumPy .npy file, in NumPy's axis order.
%
%   data = read_npy('...\pixels_dff_full.npy')
%
%   An array NumPy shows as shape (a, b, c) comes back as a MATLAB a x b x c array
%   with data(i, j, k) == numpy_array[i-1, j-1, k-1]. Supported element types:
%   float16 (decoded to double, MATLAB has no native half), float32 (single),
%   float64, int32, int64, uint8, uint16, bool. A 0-d array comes back as a scalar,
%   a 1-d array as a column.
%
%   The pixel dumps store dF/F as float16 (CLAUDE.md 9.16): 1 sign bit, 5 exponent
%   bits, 10 fraction bits. Decoded here exactly; every float16 value is exactly
%   representable in double.

fid = fopen(npy_path, 'r', 'l');
if fid < 0
    error('read_npy:open', 'Cannot open %s.', npy_path);
end
closer = onCleanup(@() fclose(fid));

magic = fread(fid, 6, 'uint8=>uint8')';
if ~isequal(magic, [147 uint8('NUMPY')])
    error('read_npy:magic', '%s is not a .npy file.', npy_path);
end
version_major = fread(fid, 1, 'uint8');
fread(fid, 1, 'uint8');                                  % minor version, unused
if version_major == 1
    header_length = fread(fid, 1, 'uint16');
else
    header_length = fread(fid, 1, 'uint32');
end
header = fread(fid, header_length, 'uint8=>char')';

descr = regexp(header, '''descr'':\s*''([^'']+)''', 'tokens', 'once');
descr = descr{1};
fortran_order = ~isempty(regexp(header, '''fortran_order'':\s*True', 'once'));
shape_text = regexp(header, '''shape'':\s*\(([^)]*)\)', 'tokens', 'once');
shape = str2double(strsplit(strtrim(shape_text{1}), ','));
shape = shape(~isnan(shape));                            % '(22,)' -> 22, '()' -> []

switch descr
    case '<f2'
        data = half_bits_to_double(fread(fid, inf, 'uint16=>double'));
    case '<f4'
        data = fread(fid, inf, 'single=>single');
    case '<f8'
        data = fread(fid, inf, 'double=>double');
    case '<i4'
        data = fread(fid, inf, 'int32=>int32');
    case '<i8'
        data = fread(fid, inf, 'int64=>int64');
    case {'|u1', '<u1'}
        data = fread(fid, inf, 'uint8=>uint8');
    case '<u2'
        data = fread(fid, inf, 'uint16=>uint16');
    case '|b1'
        data = fread(fid, inf, 'uint8=>logical');
    otherwise
        error('read_npy:dtype', '%s: element type %s is not supported.', npy_path, descr);
end

n_expected = prod(shape);                                % prod([]) == 1 for a 0-d array
if numel(data) ~= n_expected
    error('read_npy:size', '%s: %d values read, shape %s needs %d.', npy_path, numel(data), ...
        mat2str(shape), n_expected);
end
if numel(shape) <= 1
    return                                               % scalar or column
end
if fortran_order
    data = reshape(data, shape);
else
    % C order: the LAST NumPy axis varies fastest, i.e. MATLAB's first. Read it
    % reversed, then permute back to NumPy's axis order.
    data = permute(reshape(data, fliplr(shape)), numel(shape):-1:1);
end
end


function value = half_bits_to_double(bits)
% IEEE 754 binary16 bit patterns (as doubles 0..65535) -> double values.
sign = 1 - 2 * floor(bits / 32768);
exponent = mod(floor(bits / 1024), 32);
fraction = mod(bits, 1024);
value = sign .* 2 .^ (exponent - 15) .* (1 + fraction / 1024);        % normal numbers
subnormal = exponent == 0;
value(subnormal) = sign(subnormal) .* 2 ^ -14 .* fraction(subnormal) / 1024;
special = exponent == 31;
value(special & fraction == 0) = sign(special & fraction == 0) * Inf;
value(special & fraction ~= 0) = NaN;
end
