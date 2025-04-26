import { useWindowSize } from "@hooks/useWindowSize";

import Logo from "@components/Logo/Logo";

import { LogoContainerProps } from "./LogoContainer.props";

import classes from "./LogoContainer.module.css";

const LogoContainer: React.FC<LogoContainerProps> = ({
  children,
  logoOffset,
  logoSize,
  alignSelf = "center",
}) => {
  const logoStylesOfOffset = logoOffset
    ? {
        alignSelf: alignSelf,
        ...(alignSelf === "end"
          ? { marginRight: `${logoOffset}px` }
          : { marginLeft: `${logoOffset}px` }),
      }
    : undefined;
  const { isSmallMobile } = useWindowSize();
  logoSize = logoSize || isSmallMobile ? 63 : 84;

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
