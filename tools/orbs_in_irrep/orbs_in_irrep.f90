program orbs_in_irrep
! Find which orbs are in which irrep using connectivity of 1-body integrals
! If px, py are mixed, then these orbs will appear in 2 different irreps.
implicit real*8(a-h,o-z)

integer, parameter :: m_irrep=10, m_orb=200

integer irrep_orb(m_irrep,m_orb), n_orb_irrep(m_irrep)

n_irrep=0

do
  read(5,*,end=99) coef, i, j
  if(i.ne.i_prev .and. i.eq.j) then
    n_irrep=n_irrep+1
    n_orb_irrep(n_irrep)=1
    irrep_orb(n_irrep,1)=i
    i_prev=i
  endif
  if(abs(coef).lt.1e-6) cycle
! if(i.ne.i_prev .and. i.ne.j) then
  if(i.ne.j) then
    do i_irrep=1,n_irrep
      if(irrep_orb(i_irrep,1).eq.j) then
        n_orb_irrep(i_irrep)=n_orb_irrep(i_irrep)+1
        irrep_orb(i_irrep,n_orb_irrep(i_irrep))=i
        i_prev=i
!       exit
      endif
    enddo
  endif
enddo

99 continue
do i_irrep=1,n_irrep
  write(6,'(i2,2x,100i4)') i_irrep, (irrep_orb(i_irrep,i_orb), i_orb=1,n_orb_irrep(i_irrep))
enddo

stop
end
