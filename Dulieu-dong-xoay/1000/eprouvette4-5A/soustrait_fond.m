function resultat=soustrait_fond(matrice)
% resultat=soustrait_fond(matrice)
% 
% Modification des données contenues dans 'matrice' pour annuler les
% variations du fond. Cette fonction soustrait une interpolation faite à
% partir de valeurs sur les bords de la matrice, dans les deux directions.
% Si 'matrice' est à trois dimensions, le traitement est fait sur
% 'matrice(:,:,1)', 'matrice(:,:,2)', etc.
% 
% Cyril Ravat, février 2006

resultat=zeros(size(matrice));
nh=size(matrice,2);
nv=size(matrice,1);

for z=1:size(matrice,3)
    resultat(:,:,z)=matrice(:,:,z)-(ones(nv,1)*matrice(1,:,z)+(0:nv-1)'*(matrice(end,:,z)-matrice(1,:,z))/(nv-1));
    resultat(:,:,z)=resultat(:,:,z)-(resultat(:,1,z)*ones(1,nh)+(resultat(:,end,z)-resultat(:,1,z))*(0:nh-1)/(nh-1));%+mean2(matrice(:,:,z));
end
