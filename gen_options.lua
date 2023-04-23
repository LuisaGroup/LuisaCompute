function split_str(str, chr, func)
    for part in string.gmatch(str, "([^" .. chr .. "]+)") do
        func(part)
    end
end