function _config_project(config)
	if type(_configs) == "function" then
		_configs()
	end
	if type(_config_rules) == "table" then
		add_rules(_config_rules)
	end
	if type(config) == "table" then
		for k, v in pairs(config) do
			set_values(k, v)
		end
	end
	local batch_size = config["batch_size"]
	if type(batch_size) == "number" and batch_size > 1 then
		add_rules("c.unity_build", {
			batchsize = batch_size
		})
		add_rules( "c++.unity_build", {
			batchsize = batch_size
		})
	end
end
