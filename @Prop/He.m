function [] = He(prop)

prop.mg = 6.64663e-27;
prop.gamma1 = 5/3;
prop.gamma2 = @(T)(prop.gamma1+1)/(prop.gamma1-1);

end
