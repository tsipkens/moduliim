function [Tout,Ti,Cout,s_T,out] = IModel(smodel,J)

if and(strcmp(smodel.opts.pyrometry,'default'),length(smodel.l)==2)
    smodel.opts.pyrometry = '2color';
end

nshots = length(J(1,:,1)); % number of shots
ntime = length(J(:,1,1)); % number of times
switch smodel.opts.pyrometry
    
    % Two-color pyrometry **********************
    case {'2color','default'} % simple/fast two-color pyrometry, default if two wavelengths
            % cannot output scaling factor as more than one shot can be
            % used
        l = smodel.l;
        Emr = smodel.prop.Emr(l(1),l(2),smodel.prop.dp0); % two-colour pyrometry
        Tout = (0.0143877696*(1/(l(2)*1e-9)-1/(l(1)*1e-9)))./...
            log(J(:,:,1)./J(:,:,2).*(((l(1)/l(2))^6)/Emr)); % phi=0.01438
        Tout = real(Tout);
        s_T = [];
        out = [];
        Cout = [];
        
    case {'2color-scalingfactor'}
        l = smodel.l;
        Emr = smodel.prop.Emr(l(1),l(2),smodel.prop.dp0); % two-colour pyrometry
        Tout = (0.0143877696*(1/(l(2)*1e-9)-1/(l(1)*1e-9)))./...
            log(J(:,:,1)./J(:,:,2).*(((l(1)/l(2))^6)/Emr)); % phi=0.01438
        Tout = real(Tout);
        Cout = bsxfun(@times,J,1./smodel.FModel(Tout,smodel.prop.Em));
        Cout = Cout(:,:,1);
        s_T = [];
        out = [];
        
    case {'2color-constT'}
        l = smodel.l;
        Emr = smodel.prop.Emr(l(1),l(2),smodel.prop.dp0); % two-colour pyrometry
        Tout = (0.0143877696*(1/(l(2)*1e-9)-1/(l(1)*1e-9)))./...
            log(J(:,:,1)./J(:,:,2).*(((l(1)/l(2))^6)/Emr)); % phi=0.01438
        Tout = real(Tout);
        Cout = J./smodel.FModel(1730.*ones(size(Tout)),smodel.prop.Em);
        Cout = Cout(:,1,1);
        s_T = [];
        out = [];
        
    case {'2color-advanced'}  % calculate temperature, pre-averaged data
        data1 = mean(J(:,:,1),2);
        data2 = mean(J(:,:,2),2);
        [Tout,Cout] = smodel.calcRatioPyrometry(data1,data2);
        
        nn = 1000; % used for sampling methods
        s1 = std(J(:,:,1),[],2)./sqrt(nshots);
        s2 = std(J(:,:,2),[],2)./sqrt(nshots);
        datas1 = (mvnrnd(data1,s1'.^2,nn))';
        datas2 = (mvnrnd(data2,s2'.^2,nn))';
        % s_C = 0.3;
        % Cs = normrnd(1,s_C,[1,nn]);
        % datas1 = bsxfun(@times,datas1,Cs);
        % datas2 = bsxfun(@times,datas2,Cs);
        [~,~,s_T,out] = smodel.calcRatioPyrometry(datas1,datas2);
        
        out.resid = zeros(ntime,1);
        
    case {'constC'}
        if ~isfield(opts,'C')
            disp('Error occurred in pyrometry: C was not specified.');
            return
        end
        
        Tout = 1;
        s_T = [];
        out = [];
        Cout = [];
        
    % Spectral fitting / inference **********************
    otherwise
        switch smodel.opts.multicolor
            case {'constC-mass','priorC-smooth'} % simultaneous inference
                [Tout,Cout,s_T,out] = smodel.calcSpectralFit_all(J);
            otherwise % sequential inference
                [Tout,Cout,s_T,out] = smodel.calcSpectralFit(J);
        end
end

Ti = nanmean(Tout(1,:)); % average temperatures

end

