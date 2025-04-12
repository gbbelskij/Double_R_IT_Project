import useWindowWidth from "@hooks/useWindowWidth";

import Logo from "@components/Logo/Logo";

import { LogoContainerProps } from "./LogoContainer.props";

import classes from "./LogoContainer.module.css";

const LogoContainer: React.FC<LogoContainerProps> = ({
  children,
  logoOffset,
  logoSize,
}) => {
  const logoStylesOfOffset = logoOffset
    ? { alignSelf: "start", marginLeft: `${logoOffset}px` }
    : undefined;
  const windowWidth = useWindowWidth();
  logoSize = logoSize || windowWidth >= 375 ? 84 : 63;

  return (
    <div
      className={classes.LogoContainer}
      style={
        {
          "--logo-height": `${logoSize}px`,
        } as React.CSSProperties
      }
    >
      <Logo style={logoStylesOfOffset} size={logoSize} />
      {children}
    </div>
  );
};

export default LogoContainer;
